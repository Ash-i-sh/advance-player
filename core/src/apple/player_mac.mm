// macOS test player: a window with a CAMetalLayer, driven by CVDisplayLink.
//
// Each refresh: work out the playback time of the frame being prepared, ask
// the schedule for the source pair around it, interpolate on the GPU and
// present the result exactly at that refresh. Metal reports when each frame
// actually reached the screen, which is what the pacing numbers are built on.

#import <AppKit/AppKit.h>
#import <ImageIO/ImageIO.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#include <mach/mach_time.h>
#include <simd/simd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "advp/bench.h"
#include "advp/governor.h"
#include "advp/player.h"
#include "advp/rate.h"
#include "advp/schedule.h"
#include "advp_shaders.h"
#include "video_decoder.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/pixdesc.h>
}

// CVDisplayLink is deprecated from macOS 15 in favour of NSView display
// links, which need macOS 14. It still works and we support macOS 13.
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

@interface AdvpVideoView : NSView
@property(nonatomic, copy) void (^onKey)(NSEvent*);
@property(nonatomic, copy) void (^onResize)(void);
@end

@implementation AdvpVideoView
- (BOOL)acceptsFirstResponder {
  return YES;
}
- (BOOL)wantsUpdateLayer {
  return YES;
}
- (CALayer*)makeBackingLayer {
  return [CAMetalLayer layer];
}
- (void)keyDown:(NSEvent*)event {
  if (self.onKey) self.onKey(event);
}
- (void)setFrameSize:(NSSize)size {
  [super setFrameSize:size];
  if (self.onResize) self.onResize();
}
- (void)viewDidChangeBackingProperties {
  [super viewDidChangeBackingProperties];
  if (self.onResize) self.onResize();
}
@end

@interface AdvpWindowDelegate : NSObject <NSWindowDelegate>
@property(nonatomic, copy) void (^onClose)(void);
@property(nonatomic, copy) void (^onScreenChange)(void);
@end

@implementation AdvpWindowDelegate
- (void)windowWillClose:(NSNotification*)note {
  if (self.onClose) self.onClose();
}
- (void)windowDidChangeScreen:(NSNotification*)note {
  if (self.onScreenChange) self.onScreenChange();
}
@end

namespace advp {
namespace {

constexpr size_t kQueuedFrames = 8;   // decoded source frames held ahead
constexpr double kBudgetShare = 0.5;  // part of each refresh smoothing may use
constexpr double kSeekStep = 10.0;
constexpr double kWarmupSeconds = 0.5;

// Must match PresentParams in present.metal.
struct PresentParams {
  simd_float4 range;
  simd_float4 matrix;
};

double host_seconds(uint64_t host_time) {
  static const double scale = [] {
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    return static_cast<double>(info.numer) / info.denom / 1e9;
  }();
  return host_time * scale;
}

double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// YCbCr -> RGB constants for a frame, from its pixel format and colour tags.
PresentParams present_params(CVPixelBufferRef buffer) {
  const OSType format = CVPixelBufferGetPixelFormatType(buffer);
  const bool full = format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange ||
                    format == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange;
  const bool high = format == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange ||
                    format == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange;

  // 10-bit samples sit in the top bits of 16-bit texels.
  const double unit = high ? 64.0 / 65535.0 : 1.0 / 255.0;
  const double max_code = high ? 1023.0 : 255.0;
  const double mid = high ? 512.0 : 128.0;
  const double black = full ? 0.0 : (high ? 64.0 : 16.0);
  const double y_range = full ? max_code : (high ? 876.0 : 219.0);
  const double c_range = full ? max_code : (high ? 896.0 : 224.0);

  double kr = 0.2126, kb = 0.0722;  // BT.709
  CFTypeRef matrix = CVBufferCopyAttachment(buffer, kCVImageBufferYCbCrMatrixKey, nullptr);
  if (matrix) {
    if (CFEqual(matrix, kCVImageBufferYCbCrMatrix_ITU_R_601_4)) {
      kr = 0.299;
      kb = 0.114;
    } else if (CFEqual(matrix, kCVImageBufferYCbCrMatrix_ITU_R_2020)) {
      kr = 0.2627;
      kb = 0.0593;
    }
    CFRelease(matrix);
  } else if (CVPixelBufferGetHeight(buffer) <= 576) {
    kr = 0.299;  // untagged standard definition
    kb = 0.114;
  }
  const double kg = 1.0 - kr - kb;

  PresentParams p;
  p.range = simd_make_float4(black * unit, 1.0 / (y_range * unit), mid * unit,
                             1.0 / (c_range * unit));
  p.matrix = simd_make_float4(2.0 * (1.0 - kr), 2.0 * kb * (1.0 - kb) / kg,
                              2.0 * kr * (1.0 - kr) / kg, 2.0 * (1.0 - kb));
  return p;
}

// Presentation feedback, shared with Metal's presented handlers (which can
// outlive the player by a frame or two).
struct PresentStats {
  std::mutex mu;
  double period = 1.0 / 60;
  uint64_t last_generation = 0;
  double last_time = 0;
  int64_t presented = 0;
  int64_t missed = 0;
  std::vector<double> errors;  // actual minus expected time since the previous frame

  // expected: how long after the previous frame this one should appear.
  void record(uint64_t generation, double expected, double time) {
    std::lock_guard<std::mutex> lock(mu);
    if (time <= 0) {
      ++missed;  // the drawable never reached the screen
      return;
    }
    ++presented;
    if (generation == last_generation && last_time > 0) {
      const double interval = time - last_time;
      if (interval > 0 && interval < 0.25) {
        errors.push_back(interval - expected);
        missed += std::max<long>(0, std::lround((interval - expected) / period));
      }
    }
    last_generation = generation;
    last_time = time;
  }
};

class Player {
  // An engine instance and the source pair it has analysed.
  struct Slot {
    std::unique_ptr<Interpolator> engine;
    GpuFramePtr a;
    GpuFramePtr b;
    uint64_t last_used = 0;
  };

 public:
  Player(GpuBackend& backend, const PlayOptions& options)
      : backend_(backend), opt_(options), stats_(std::make_shared<PresentStats>()) {}

  ~Player() {
    if (link_ || decode_thread_.joinable()) shutdown();
  }

  PlayResult run() {
    const auto started = std::chrono::steady_clock::now();
    decoder_ = std::make_unique<VideoDecoder>(backend_, opt_.input);
    const AVCodecParameters* par = decoder_->video_stream()->codecpar;
    result_.width = par->width;
    result_.height = par->height;
    result_.source_fps = decoder_->frame_rate().value();
    const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(par->format));
    high_bit_depth_ = desc && desc->comp[0].depth > 8;
    const AVRational sar = par->sample_aspect_ratio;
    aspect_ = static_cast<double>(par->width) / par->height *
              (sar.num > 0 && sar.den > 0 ? av_q2d(sar) : 1.0);

    [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
    setup_metal();
    open_window();
    open_display_link();
    choose_output_rate();
    choose_engines();

    decode_thread_ = std::thread([this] { decode_loop(); });
    CVDisplayLinkStart(link_);

    NSTimer* title_timer = [NSTimer scheduledTimerWithTimeInterval:0.5
                                                           repeats:YES
                                                             block:^(NSTimer*) {
                                                               update_title();
                                                             }];
    NSTimer* limit_timer = nil;
    if (opt_.max_seconds > 0) {
      limit_timer = [NSTimer scheduledTimerWithTimeInterval:opt_.max_seconds
                                                    repeats:NO
                                                      block:^(NSTimer*) {
                                                        request_stop();
                                                      }];
    }
    [NSApp activateIgnoringOtherApps:YES];
    [NSApp run];

    [title_timer invalidate];
    [limit_timer invalidate];
    shutdown();
    result_.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    finish_result();
    return result_;
  }

 private:
  // ---- setup ---------------------------------------------------------------

  void setup_metal() {
    device_ = MTLCreateSystemDefaultDevice();
    queue_ = [device_ newCommandQueue];
    NSError* error = nil;
    id<MTLLibrary> library = [device_ newLibraryWithSource:@(kPresentShaderSource)
                                                   options:[MTLCompileOptions new]
                                                     error:&error];
    if (!library) {
      throw std::runtime_error(std::string("present shader: ") +
                               error.localizedDescription.UTF8String);
    }
    MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = [library newFunctionWithName:@"present_vertex"];
    d.fragmentFunction = [library newFunctionWithName:@"present_fragment"];
    d.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    pipeline_ = [device_ newRenderPipelineStateWithDescriptor:d error:&error];
    if (!pipeline_) throw std::runtime_error("cannot build present pipeline");
    if (CVMetalTextureCacheCreate(kCFAllocatorDefault, nullptr, device_, nullptr, &cache_) !=
        kCVReturnSuccess) {
      throw std::runtime_error("cannot create Metal texture cache");
    }
  }

  void open_window() {
    NSScreen* screen = NSScreen.mainScreen;
    const CGFloat scale = screen.backingScaleFactor;
    NSSize size = NSMakeSize(result_.height * aspect_ / scale, result_.height / scale);
    const NSRect visible = screen.visibleFrame;
    const CGFloat fit = std::min({1.0, visible.size.width * 0.8 / size.width,
                                  visible.size.height * 0.8 / size.height});
    size = NSMakeSize(std::floor(size.width * fit), std::floor(size.height * fit));

    window_ = [[NSWindow alloc]
        initWithContentRect:NSMakeRect(0, 0, size.width, size.height)
                  styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                            NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable
                    backing:NSBackingStoreBuffered
                      defer:NO];
    window_.releasedWhenClosed = NO;
    window_.collectionBehavior = NSWindowCollectionBehaviorFullScreenPrimary;
    window_.backgroundColor = NSColor.blackColor;
    window_.contentAspectRatio = size;
    [window_ center];

    view_ = [[AdvpVideoView alloc] initWithFrame:NSMakeRect(0, 0, size.width, size.height)];
    view_.wantsLayer = YES;
    layer_ = static_cast<CAMetalLayer*>(view_.layer);
    layer_.device = device_;
    layer_.pixelFormat = MTLPixelFormatBGRA8Unorm;
    layer_.framebufferOnly = YES;
    layer_.displaySyncEnabled = YES;
    layer_.maximumDrawableCount = 3;
    CGColorSpaceRef space = CGColorSpaceCreateWithName(kCGColorSpaceITUR_709);
    layer_.colorspace = space;
    CGColorSpaceRelease(space);

    Player* self = this;
    view_.onKey = ^(NSEvent* event) {
      self->handle_key(event);
    };
    view_.onResize = ^{
      self->resize_layer();
    };
    delegate_ = [AdvpWindowDelegate new];
    delegate_.onClose = ^{
      self->request_stop();
    };
    delegate_.onScreenChange = ^{
      self->follow_screen();
    };
    window_.delegate = delegate_;
    window_.contentView = view_;
    [window_ makeFirstResponder:view_];
    resize_layer();
    [window_ makeKeyAndOrderFront:nil];
    if (opt_.fullscreen) [window_ toggleFullScreen:nil];
  }

  void resize_layer() {
    const CGFloat scale = window_.backingScaleFactor;
    layer_.contentsScale = scale;
    layer_.drawableSize = CGSizeMake(view_.bounds.size.width * scale,
                                     view_.bounds.size.height * scale);
  }

  void open_display_link() {
    if (CVDisplayLinkCreateWithActiveCGDisplays(&link_) != kCVReturnSuccess) {
      throw std::runtime_error("cannot create display link");
    }
    CVDisplayLinkSetOutputCallback(link_, &Player::display_callback, this);
    follow_screen();
  }

  // Keeps the display link on the screen showing the window. The display
  // link's own period is only valid once it runs, so start from the screen's
  // rate and let on_refresh() correct it.
  void follow_screen() {
    NSScreen* screen = window_.screen ?: NSScreen.mainScreen;
    NSNumber* id = screen.deviceDescription[@"NSScreenNumber"];
    if (id) CVDisplayLinkSetCurrentCGDisplay(link_, id.unsignedIntValue);
    const NSInteger fps = screen.maximumFramesPerSecond;
    set_refresh_period(1.0 / (fps > 0 ? fps : 60));
  }

  void set_refresh_period(double seconds) {
    display_hz_ = 1.0 / seconds;
    std::lock_guard<std::mutex> lock(stats_->mu);
    stats_->period = seconds;
  }

  // "display", "fit:N" (highest rate up to N that divides the refresh evenly)
  // or a fixed rate as accepted by resolve_target_fps.
  void choose_output_rate() {
    const std::string& fps = opt_.fps;
    const double hz = display_hz_;
    if (fps.empty() || fps == "display") {
      output_fps_ = 0;
    } else if (fps.rfind("fit:", 0) == 0) {
      const double cap = std::atof(fps.c_str() + 4);
      if (cap <= 0) throw std::runtime_error("bad --fps " + fps);
      output_fps_ = hz / std::max(1.0, std::ceil(hz / cap - 1e-6));
    } else {
      const auto r = resolve_target_fps(fps.c_str(), decoder_->frame_rate());
      if (!r) throw std::runtime_error("bad --fps " + fps);
      output_fps_ = r->value();
    }
    // Rates within 0.5% of the refresh are treated as the refresh itself.
    if (output_fps_ > 0 && std::fabs(output_fps_ / hz - 1.0) < 0.005) output_fps_ = 0;
    const double rate = output_fps_ > 0 ? output_fps_ : hz;
    const double hold = hz / rate;
    result_.output_fps = rate;
    result_.even_cadence = std::fabs(hold - std::round(hold)) < 0.01 && hold >= 0.99;
    result_.even_fps = hz / std::max(1.0, std::ceil(hz / rate - 1e-6));
  }

  void choose_engines() {
    const double out_fps = output_fps_ > 0 ? output_fps_ : display_hz_.load();
    if (opt_.engine == "auto") {
      // Measure here, at this video's size, rather than trusting a device list.
      const auto bench = benchmark_engines(backend_, result_.width, result_.height, 12);
      const auto selection = select_engine(bench, result_.source_fps, out_fps, kBudgetShare);
      governor_.emplace(selection.ladder, 0);
      set_engine(governor_->current());
    } else {
      set_engine(opt_.engine);
    }
    result_.start_engine = engine_name_;
  }

  void set_engine(const std::string& name) {
    for (auto& slot : slots_) {
      slot = Slot{};
      if (name == "off") continue;
      slot.engine = backend_.create_engine(name);
      // Run once on blank frames so textures and buffer pools exist before
      // the first real refresh needs them.
      auto a = backend_.new_frame(result_.width, result_.height, high_bit_depth_);
      auto b = backend_.new_frame(result_.width, result_.height, high_bit_depth_);
      auto out = backend_.new_frame_like(*a);
      slot.engine->set_pair(a, b);
      slot.engine->interpolate(0.5f, *out);
    }
    std::lock_guard<std::mutex> lock(mu_);
    engine_name_ = name;
  }

  // ---- decoding (own thread) ------------------------------------------------

  void decode_loop() {
    try {
      const double step = 1.0 / decoder_->frame_rate().value();
      double discard_before = -1e300;
      if (opt_.start > 0) {
        decoder_->seek(opt_.start);
        discard_before = opt_.start - step / 2;
      }
      DecodedFrame decoded;
      while (!stop_) {
        std::optional<double> seek_to;
        {
          std::unique_lock<std::mutex> lock(mu_);
          cv_.wait(lock, [&] {
            return stop_ || seek_request_ || (!schedule_.ended() && schedule_.size() < kQueuedFrames);
          });
          if (stop_) break;
          if (seek_request_) {
            seek_to = seek_request_;
            seek_request_.reset();
            schedule_.clear();
            clock_running_ = false;
            restart_media_.reset();
          }
        }
        if (seek_to) {
          decoder_->seek(*seek_to);
          discard_before = *seek_to - step / 2;
          continue;
        }
        const bool got = decoder_->next(decoded);
        std::lock_guard<std::mutex> lock(mu_);
        if (seek_request_) continue;  // stale: a seek arrived while decoding
        if (!got) {
          schedule_.end();
        } else if (decoded.time >= discard_before) {
          schedule_.push(std::move(decoded.frame), decoded.time);
          hardware_ = decoder_->hardware();
        }
        decoded.frame.reset();
      }
    } catch (const std::exception& e) {
      {
        std::lock_guard<std::mutex> lock(mu_);
        error_ = e.what();
      }
      request_stop();
    }
  }

  // ---- display (display link thread) ----------------------------------------

  static CVReturn display_callback(CVDisplayLinkRef, const CVTimeStamp*, const CVTimeStamp* out,
                                   CVOptionFlags, CVOptionFlags*, void* context) {
    auto* self = static_cast<Player*>(context);
    self->in_callback_ = true;
    if (!self->stop_) {
      @autoreleasepool {
        try {
          self->on_refresh(out);
        } catch (const std::exception& e) {
          {
            std::lock_guard<std::mutex> lock(self->mu_);
            self->error_ = e.what();
          }
          self->request_stop();
        }
      }
    }
    self->in_callback_ = false;
    return kCVReturnSuccess;
  }

  void on_refresh(const CVTimeStamp* out) {
    const double present_at = host_seconds(out->hostTime);
    double period = 1.0 / display_hz_;
    if (out->videoTimeScale > 0 && out->videoRefreshPeriod > 0) {
      period = static_cast<double>(out->videoRefreshPeriod) / out->videoTimeScale;
      if (std::fabs(period * display_hz_ - 1.0) > 0.01) set_refresh_period(period);
    }

    double media = 0;
    bool playing = false;
    uint64_t generation = 0;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (!clock_running_ && !paused_) {
        // Start (or restart after a seek) once the next frames are decoded.
        if (schedule_.size() >= 2 || (schedule_.ended() && !schedule_.empty())) {
          clock_running_ = true;
          origin_host_ = present_at;
          origin_media_ = restart_media_.value_or(schedule_.first_time());
          restart_media_.reset();
          ++generation_;
        }
      }
      if (!clock_running_ && !paused_) return;
      media = paused_ ? paused_media_ : origin_media_ + (present_at - origin_host_);
      current_media_ = media;
      playing = !paused_;
      generation = generation_;
    }
    if (playing) {
      ++refreshes_;
      check_thermal();
    }
    if (output_fps_ > 0) {
      fixed_rate_refresh(media, present_at, period, playing, generation);
    } else {
      display_rate_refresh(media, present_at, period, playing, generation);
    }
  }

  // Display rate: a new frame for every refresh, made during that refresh.
  void display_rate_refresh(double media, double present_at, double period, bool playing,
                            uint64_t generation) {
    const ScheduledPair pick = pick_at(media);
    if (pick.finished) {
      request_stop();
      return;
    }
    if (!pick.a) return;
    if (playing && pick.underflow) ++underflows_;

    double work_ms = 0;
    const GpuFramePtr frame = render(pick, playing, &work_ms);
    report_work(work_ms, period * 1000.0, playing);
    draw(frame, present_at,
         playing ? std::optional<Measure>(Measure{generation, period}) : std::nullopt);
    maybe_screenshot(frame, pick, present_at, playing);
    analyse_next(pick);
  }

  // Fixed rate: output frame k belongs at origin + k / fps and stays on
  // screen until frame k+1 is due. Each frame is made once, ahead of time,
  // while the previous one is showing, so it gets the whole gap as budget.
  void fixed_rate_refresh(double media, double present_at, double period, bool playing,
                          uint64_t generation) {
    if (generation != out_generation_) {
      out_generation_ = generation;
      shown_slot_ = -1;
      ready_.reset();
    }
    const int64_t slot = static_cast<int64_t>(std::floor((media - origin_media_) * output_fps_ + 1e-6));
    if (slot != shown_slot_) {
      const ScheduledPair pick = pick_at(slot_time(slot));
      if (pick.finished) {
        request_stop();
        return;
      }
      if (!pick.a) return;
      if (playing && pick.underflow) ++underflows_;

      GpuFramePtr frame;
      if (ready_ && ready_->slot == slot) {
        frame = ready_->frame;
      } else {
        double work_ms = 0;
        frame = render(pick, playing, &work_ms);
        report_work(work_ms, period * 1000.0, playing);
      }
      ready_.reset();
      const double expected = slot_host(slot, period) - slot_host(slot - 1, period);
      draw(frame, present_at,
           playing && shown_slot_ >= 0 && slot == shown_slot_ + 1
               ? std::optional<Measure>(Measure{generation, expected})
               : std::nullopt);
      shown_slot_ = slot;
      maybe_screenshot(frame, pick, present_at, playing);
    }

    // Make the next frame now, while this one is on screen.
    if (!ready_ && shown_slot_ >= 0) {
      const int64_t next = shown_slot_ + 1;
      const ScheduledPair pick = pick_at(slot_time(next));
      // Only once both of its source frames are decoded; otherwise it is
      // made on its own refresh.
      if (pick.a && !pick.finished && (pick.b != pick.a || pick.t == 0.f) && !pick.underflow) {
        double work_ms = 0;
        GpuFramePtr frame = render(pick, playing, &work_ms);
        const double gap = std::max(period, slot_host(next, period) - present_at);
        report_work(work_ms, gap * 1000.0, playing);
        ready_ = Ready{next, std::move(frame)};
        analyse_next(pick);
      }
    }
  }

  double slot_time(int64_t slot) const { return origin_media_ + slot / output_fps_; }

  // The refresh a slot is shown on: the first one at or after its time.
  double slot_host(int64_t slot, double period) const {
    const double ideal = slot / output_fps_;
    return origin_host_ + std::ceil(ideal / period - 1e-6) * period;
  }

  ScheduledPair pick_at(double time) {
    ScheduledPair pick;
    {
      std::lock_guard<std::mutex> lock(mu_);
      pick = schedule_.at(time);
    }
    cv_.notify_all();  // the schedule may have room for more frames now
    return pick;
  }

  // The frame for pick: a source frame, or an interpolated one.
  GpuFramePtr render(const ScheduledPair& pick, bool playing, double* work_ms) {
    GpuFramePtr frame = pick.t >= 1.f ? pick.b : pick.a;
    if (smoothing_ && slots_[0].engine && pick.t > 0.f && pick.t < 1.f) {
      const double t0 = now_ms();
      Slot& slot = slot_for(pick.a, pick.b);
      frame = backend_.new_frame_like(*pick.a);
      slot.engine->interpolate(pick.t, *frame);
      *work_ms = now_ms() - t0;
      if (playing) {
        ++interpolated_;
        work_ms_.push_back(*work_ms);
      }
    }
    return frame;
  }

  // window_ms: time this frame had before it was due. Load is judged over a
  // source frame's worth of output, because motion analysis lands on one
  // frame per source pair; a frame that overran its own window counts at once.
  void report_work(double work_ms, double window_ms, bool playing) {
    recent_work_ms_ = recent_work_ms_ * 0.95 + work_ms * 0.05;
    if (!governor_ || !playing || !smoothing_) return;
    // The first frames after a start or seek have no work done ahead; judge
    // the engine on steady playback.
    if (current_media_ - origin_media_ < kWarmupSeconds) return;
    if (work_ms > window_ms) {
      governor_->report(work_ms, window_ms * kBudgetShare);
      work_sum_ = window_sum_ = 0;
    } else {
      work_sum_ += work_ms;
      window_sum_ += window_ms;
      if (window_sum_ < 1000.0 / result_.source_fps) return;
      governor_->report(work_sum_, window_sum_ * kBudgetShare);
      work_sum_ = window_sum_ = 0;
    }
    if (governor_->take_changed()) {
      set_engine(governor_->current());
      ++engine_changes_;
    }
  }

  // Queue motion analysis of the next pair now, so the GPU does it in the
  // idle time before the refresh that needs it.
  void analyse_next(const ScheduledPair& pick) {
    if (smoothing_ && slots_[0].engine && pick.c && pick.b != pick.a) slot_for(pick.b, pick.c);
  }

  void maybe_screenshot(const GpuFramePtr& frame, const ScheduledPair& pick, double present_at,
                        bool playing) {
    if (opt_.screenshot.empty() || screenshot_taken_ || !playing || pick.t <= 0.f ||
        pick.t >= 1.f || present_at - origin_host_ < opt_.screenshot_after) {
      return;
    }
    screenshot_taken_ = true;
    save_png(frame, opt_.screenshot);
  }

  // The engine instance analysing (a, b), starting the analysis if no
  // instance has it. Analysis is queued on the GPU without waiting.
  Slot& slot_for(const GpuFramePtr& a, const GpuFramePtr& b) {
    for (auto& slot : slots_) {
      if (slot.a == a && slot.b == b) {
        slot.last_used = ++slot_clock_;
        return slot;
      }
    }
    Slot& slot = slots_[0].last_used <= slots_[1].last_used ? slots_[0] : slots_[1];
    slot.engine->set_pair(a, b);
    slot.a = a;
    slot.b = b;
    slot.last_used = ++slot_clock_;
    return slot;
  }

  // Thermal pressure: give up a quality step before the system throttles us.
  void check_thermal() {
    if (!governor_ || refreshes_ % 120 != 0) return;
    const NSProcessInfoThermalState state = NSProcessInfo.processInfo.thermalState;
    if (state >= NSProcessInfoThermalStateSerious && last_thermal_ < NSProcessInfoThermalStateSerious) {
      governor_->force_step_down();
      if (governor_->take_changed()) {
        set_engine(governor_->current());
        ++engine_changes_;
      }
    }
    last_thermal_ = state;
  }

  struct Measure {
    uint64_t generation;
    double expected;  // seconds after the previous frame
  };

  void draw(const GpuFramePtr& frame, double present_at, std::optional<Measure> measure) {
    id<CAMetalDrawable> drawable = [layer_ nextDrawable];
    if (!drawable) return;

    auto buffer = static_cast<CVPixelBufferRef>(frame->native_handle());
    const bool high = frame->high_bit_depth();
    CVMetalTextureRef y_ref = nullptr;
    CVMetalTextureRef uv_ref = nullptr;
    CVMetalTextureCacheCreateTextureFromImage(
        kCFAllocatorDefault, cache_, buffer, nullptr,
        high ? MTLPixelFormatR16Unorm : MTLPixelFormatR8Unorm, CVPixelBufferGetWidthOfPlane(buffer, 0),
        CVPixelBufferGetHeightOfPlane(buffer, 0), 0, &y_ref);
    CVMetalTextureCacheCreateTextureFromImage(
        kCFAllocatorDefault, cache_, buffer, nullptr,
        high ? MTLPixelFormatRG16Unorm : MTLPixelFormatRG8Unorm, CVPixelBufferGetWidthOfPlane(buffer, 1),
        CVPixelBufferGetHeightOfPlane(buffer, 1), 1, &uv_ref);
    if (!y_ref || !uv_ref) {
      if (y_ref) CFRelease(y_ref);
      if (uv_ref) CFRelease(uv_ref);
      throw std::runtime_error("cannot map frame for display");
    }
    id<MTLCommandBuffer> cb = [queue_ commandBuffer];
    encode_present(cb, buffer, y_ref, uv_ref, drawable.texture);

    if (measure) {
      std::shared_ptr<PresentStats> stats = stats_;
      const Measure m = *measure;
      [drawable addPresentedHandler:^(id<MTLDrawable> shown) {
        stats->record(m.generation, m.expected, shown.presentedTime);
      }];
    }
    [cb presentDrawable:drawable atTime:present_at];
    GpuFramePtr keep = frame;  // the GPU reads it until the buffer completes
    [cb addCompletedHandler:^(id<MTLCommandBuffer>) {
      CFRelease(y_ref);
      CFRelease(uv_ref);
      (void)keep;
    }];
    [cb commit];
  }

  // YCbCr -> RGB into target, letterboxed to the video's display aspect.
  void encode_present(id<MTLCommandBuffer> cb, CVPixelBufferRef buffer, CVMetalTextureRef y_ref,
                      CVMetalTextureRef uv_ref, id<MTLTexture> target) {
    const PresentParams params = present_params(buffer);
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;

    const double dw = target.width;
    const double dh = target.height;
    double vw = dw, vh = dw / aspect_;
    if (vh > dh) {
      vh = dh;
      vw = dh * aspect_;
    }
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:pass];
    [enc setViewport:MTLViewport{(dw - vw) / 2, (dh - vh) / 2, vw, vh, 0, 1}];
    [enc setRenderPipelineState:pipeline_];
    [enc setFragmentTexture:CVMetalTextureGetTexture(y_ref) atIndex:0];
    [enc setFragmentTexture:CVMetalTextureGetTexture(uv_ref) atIndex:1];
    [enc setFragmentBytes:&params length:sizeof params atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
  }

  // Renders frame exactly as displayed, at the video's own size, to a PNG.
  void save_png(const GpuFramePtr& frame, const std::string& path) {
    auto buffer = static_cast<CVPixelBufferRef>(frame->native_handle());
    const bool high = frame->high_bit_depth();
    CVMetalTextureRef y_ref = nullptr;
    CVMetalTextureRef uv_ref = nullptr;
    CVMetalTextureCacheCreateTextureFromImage(
        kCFAllocatorDefault, cache_, buffer, nullptr,
        high ? MTLPixelFormatR16Unorm : MTLPixelFormatR8Unorm, CVPixelBufferGetWidthOfPlane(buffer, 0),
        CVPixelBufferGetHeightOfPlane(buffer, 0), 0, &y_ref);
    CVMetalTextureCacheCreateTextureFromImage(
        kCFAllocatorDefault, cache_, buffer, nullptr,
        high ? MTLPixelFormatRG16Unorm : MTLPixelFormatRG8Unorm, CVPixelBufferGetWidthOfPlane(buffer, 1),
        CVPixelBufferGetHeightOfPlane(buffer, 1), 1, &uv_ref);
    if (!y_ref || !uv_ref) {
      if (y_ref) CFRelease(y_ref);
      if (uv_ref) CFRelease(uv_ref);
      return;
    }
    const int w = static_cast<int>(std::lround(result_.height * aspect_));
    const int h = result_.height;
    MTLTextureDescriptor* d =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                           width:w
                                                          height:h
                                                       mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget;
    d.storageMode = MTLStorageModeShared;
    id<MTLTexture> target = [device_ newTextureWithDescriptor:d];
    id<MTLCommandBuffer> cb = [queue_ commandBuffer];
    encode_present(cb, buffer, y_ref, uv_ref, target);
    [cb commit];
    [cb waitUntilCompleted];
    CFRelease(y_ref);
    CFRelease(uv_ref);

    std::vector<uint8_t> pixels(static_cast<size_t>(w) * h * 4);
    [target getBytes:pixels.data() bytesPerRow:w * 4 fromRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0];
    CGColorSpaceRef space = CGColorSpaceCreateWithName(kCGColorSpaceITUR_709);
    CGContextRef ctx = CGBitmapContextCreate(pixels.data(), w, h, 8, w * 4, space,
                                             static_cast<uint32_t>(kCGImageAlphaNoneSkipFirst) |
                                                 kCGBitmapByteOrder32Little);
    CGImageRef image = CGBitmapContextCreateImage(ctx);
    NSURL* url = [NSURL fileURLWithPath:@(path.c_str())];
    CGImageDestinationRef dest = CGImageDestinationCreateWithURL(
        (__bridge CFURLRef)url, (__bridge CFStringRef)UTTypePNG.identifier, 1, nullptr);
    if (dest) {
      CGImageDestinationAddImage(dest, image, nullptr);
      CGImageDestinationFinalize(dest);
      CFRelease(dest);
    }
    CGImageRelease(image);
    CGContextRelease(ctx);
    CGColorSpaceRelease(space);
  }

  // ---- controls (main thread) -----------------------------------------------

  void handle_key(NSEvent* event) {
    const unsigned short code = event.keyCode;
    NSString* chars = event.charactersIgnoringModifiers.lowercaseString;
    if (code == 53 || [chars isEqualToString:@"q"]) {
      request_stop();
    } else if ([chars isEqualToString:@" "]) {
      toggle_pause();
    } else if ([chars isEqualToString:@"s"]) {
      smoothing_ = !smoothing_;
    } else if ([chars isEqualToString:@"f"]) {
      [window_ toggleFullScreen:nil];
    } else if (code == 123 || code == 124) {
      seek_by(code == 123 ? -kSeekStep : kSeekStep);
    }
    update_title();
  }

  void toggle_pause() {
    std::lock_guard<std::mutex> lock(mu_);
    if (paused_) {
      paused_ = false;
      clock_running_ = false;
      restart_media_ = paused_media_;
    } else if (clock_running_) {
      paused_ = true;
      paused_media_ = current_media_;
    }
  }

  void seek_by(double delta) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      double target = std::max(0.0, current_media_ + delta);
      if (decoder_->duration() > 1) target = std::min(target, decoder_->duration() - 1);
      seek_request_ = target;
      paused_ = false;
    }
    cv_.notify_all();
  }

  void update_title() {
    std::string engine;
    double media = 0;
    bool paused = false;
    {
      std::lock_guard<std::mutex> lock(mu_);
      engine = engine_name_;
      media = current_media_;
      paused = paused_;
    }
    int64_t missed = 0;
    {
      std::lock_guard<std::mutex> lock(stats_->mu);
      missed = stats_->missed;
    }
    NSString* name = [@(opt_.input.c_str()) lastPathComponent];
    window_.title = [NSString
        stringWithFormat:@"%@ — %s%s · %.3f → %.2f fps on %.0f Hz · GPU %.1f ms · missed %lld · %.0f s%s",
                         name, smoothing_ ? engine.c_str() : "off",
                         smoothing_ ? "" : " (S)", result_.source_fps,
                         output_fps_ > 0 ? output_fps_ : display_hz_.load(),
                         display_hz_.load(), recent_work_ms_.load(), static_cast<long long>(missed), media,
                         paused ? " · paused" : ""];
  }

  // ---- shutdown -------------------------------------------------------------

  void request_stop() {
    if (stop_.exchange(true)) return;
    cv_.notify_all();
    dispatch_async(dispatch_get_main_queue(), ^{
      [NSApp stop:nil];
      // stop: takes effect after the next event, so post one.
      [NSApp postEvent:[NSEvent otherEventWithType:NSEventTypeApplicationDefined
                                          location:NSZeroPoint
                                     modifierFlags:0
                                         timestamp:0
                                      windowNumber:0
                                           context:nil
                                           subtype:0
                                             data1:0
                                             data2:0]
               atStart:NO];
    });
  }

  void shutdown() {
    stop_ = true;
    cv_.notify_all();
    if (link_) {
      CVDisplayLinkStop(link_);
      while (in_callback_) std::this_thread::sleep_for(std::chrono::milliseconds(1));
      CVDisplayLinkRelease(link_);
      link_ = nullptr;
    }
    if (decode_thread_.joinable()) decode_thread_.join();
    id<MTLCommandBuffer> idle = [queue_ commandBuffer];
    [idle commit];
    [idle waitUntilCompleted];
    delegate_.onClose = nil;
    [window_ close];
    if (cache_) {
      CFRelease(cache_);
      cache_ = nullptr;
    }
  }

  void finish_result() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      result_.error = error_;
      result_.final_engine = engine_name_;
    }
    result_.ok = result_.error.empty();
    result_.decoder = hardware_ ? "hardware" : "software";
    result_.display_hz = display_hz_;
    result_.engine_changes = engine_changes_;
    result_.refreshes = refreshes_;
    result_.interpolated = interpolated_;
    result_.underflows = underflows_;

    if (!work_ms_.empty()) {
      std::vector<double> w = work_ms_;
      std::sort(w.begin(), w.end());
      double sum = 0;
      for (double v : w) sum += v;
      result_.work_ms_mean = sum / w.size();
      result_.work_ms_p99 = w[std::min(w.size() - 1, w.size() * 99 / 100)];
      result_.work_ms_max = w.back();
    }
    std::lock_guard<std::mutex> lock(stats_->mu);
    result_.presented = stats_->presented;
    result_.missed = stats_->missed;
    const auto& err = stats_->errors;
    if (!err.empty()) {
      double sq = 0;
      for (double v : err) sq += v * v;
      result_.interval_jitter_ms = std::sqrt(sq / err.size()) * 1000.0;
    }
  }

  GpuBackend& backend_;
  PlayOptions opt_;
  PlayResult result_;
  std::unique_ptr<VideoDecoder> decoder_;
  double aspect_ = 16.0 / 9.0;
  bool high_bit_depth_ = false;

  // Window and presentation.
  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  id<MTLRenderPipelineState> pipeline_ = nil;
  CVMetalTextureCacheRef cache_ = nullptr;
  NSWindow* window_ = nil;
  AdvpVideoView* view_ = nil;
  CAMetalLayer* layer_ = nil;
  AdvpWindowDelegate* delegate_ = nil;
  CVDisplayLinkRef link_ = nullptr;
  std::shared_ptr<PresentStats> stats_;

  // Shared between threads, guarded by mu_.
  std::mutex mu_;
  std::condition_variable cv_;
  FrameSchedule schedule_;
  bool clock_running_ = false;
  double origin_host_ = 0;
  double origin_media_ = 0;
  std::optional<double> restart_media_;
  double current_media_ = 0;
  bool paused_ = false;
  double paused_media_ = 0;
  std::optional<double> seek_request_;
  uint64_t generation_ = 0;  // bumps on every clock restart; splits timing stats
  std::string engine_name_;
  std::string error_;
  bool hardware_ = false;

  std::atomic<bool> stop_{false};
  std::atomic<bool> in_callback_{false};
  std::atomic<bool> smoothing_{true};
  std::atomic<double> recent_work_ms_{0};
  std::atomic<double> display_hz_{60};
  std::thread decode_thread_;

  // Display thread only.
  Slot slots_[2];
  uint64_t slot_clock_ = 0;
  std::optional<Governor> governor_;
  NSProcessInfoThermalState last_thermal_ = NSProcessInfoThermalStateNominal;
  int engine_changes_ = 0;
  int64_t refreshes_ = 0;
  int64_t interpolated_ = 0;
  int64_t underflows_ = 0;
  std::vector<double> work_ms_;
  bool screenshot_taken_ = false;

  // Fixed output rate (0: one frame per display refresh).
  struct Ready {
    int64_t slot;
    GpuFramePtr frame;
  };
  double output_fps_ = 0;
  double work_sum_ = 0;
  double window_sum_ = 0;
  uint64_t out_generation_ = 0;
  int64_t shown_slot_ = -1;
  std::optional<Ready> ready_;
};

}  // namespace

PlayResult play_file(GpuBackend& backend, const PlayOptions& options) {
  try {
    Player player(backend, options);
    return player.run();
  } catch (const std::exception& e) {
    PlayResult r;
    r.error = e.what();
    return r;
  }
}

}  // namespace advp
