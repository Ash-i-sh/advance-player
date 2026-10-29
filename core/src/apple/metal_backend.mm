// Metal backend: frames are IOSurface-backed CVPixelBuffers, shared zero-copy
// between VideoToolbox decode, Metal compute and VideoToolbox encode.

#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <simd/simd.h>

#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "advp/engine.h"
#include "advp_shaders.h"

namespace advp {
namespace {

// Must match the structs in interpolate.metal.
struct SearchParams {
  simd_int2 field_size;
  simd_int2 coarse_field_size;
  int radius;
  float lambda;
  int refine_iters;
  int subpel;
  float bad_sad;
  int count_stats;
  float new_penalty;
};

struct InterpParams {
  float t;
  int mode;
  float cut_fraction;
  uint32_t total_blocks;
  float occlusion_lo;
  float occlusion_hi;
  int bicubic;
  float candidate_penalty;
  float score_scale;
};

bool is_high_bit_depth(OSType format) {
  return format == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange ||
         format == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange;
}

bool is_supported(OSType format) {
  return format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange ||
         format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange || is_high_bit_depth(format);
}

int blocks(int pixels) { return (pixels + 7) / 8; }


// Luma mismatch (0..1) charged to the zero-motion and raw-block-vector
// candidates, so they replace the smooth field only where clearly better.
// Measured: 0.004 cost pans 0.4 dB; 0.03-0.06 is the plateau on both clips.
constexpr float kCandidatePenalty = 0.04f;

class MetalFrame final : public GpuFrame {
 public:
  MetalFrame(CVPixelBufferRef buffer, CVMetalTextureCacheRef cache)
      : buffer_(CVPixelBufferRetain(buffer)) {
    const OSType format = CVPixelBufferGetPixelFormatType(buffer);
    if (!is_supported(format)) {
      CVPixelBufferRelease(buffer_);
      throw std::runtime_error("unsupported pixel format for GPU processing");
    }
    high_ = is_high_bit_depth(format);
    y_ = plane(cache, 0, high_ ? MTLPixelFormatR16Unorm : MTLPixelFormatR8Unorm, &y_ref_);
    uv_ = plane(cache, 1, high_ ? MTLPixelFormatRG16Unorm : MTLPixelFormatRG8Unorm, &uv_ref_);
  }

  ~MetalFrame() override {
    if (y_ref_) CFRelease(y_ref_);
    if (uv_ref_) CFRelease(uv_ref_);
    CVPixelBufferRelease(buffer_);
  }

  int width() const override { return static_cast<int>(CVPixelBufferGetWidth(buffer_)); }
  int height() const override { return static_cast<int>(CVPixelBufferGetHeight(buffer_)); }
  bool high_bit_depth() const override { return high_; }
  void* native_handle() const override { return buffer_; }

  CVPixelBufferRef buffer() const { return buffer_; }
  id<MTLTexture> y() const { return y_; }
  id<MTLTexture> uv() const { return uv_; }

 private:
  id<MTLTexture> plane(CVMetalTextureCacheRef cache, size_t index, MTLPixelFormat format,
                       CVMetalTextureRef* ref) {
    const size_t w = CVPixelBufferGetWidthOfPlane(buffer_, index);
    const size_t h = CVPixelBufferGetHeightOfPlane(buffer_, index);
    const CVReturn rc = CVMetalTextureCacheCreateTextureFromImage(
        kCFAllocatorDefault, cache, buffer_, nullptr, format, w, h, index, ref);
    if (rc != kCVReturnSuccess || !*ref) {
      throw std::runtime_error("cannot map frame plane to a Metal texture");
    }
    return CVMetalTextureGetTexture(*ref);
  }

  CVPixelBufferRef buffer_;
  CVMetalTextureRef y_ref_ = nullptr;
  CVMetalTextureRef uv_ref_ = nullptr;
  id<MTLTexture> y_;
  id<MTLTexture> uv_;
  bool high_ = false;
};

const MetalFrame& as_metal(const GpuFrame& frame) {
  return static_cast<const MetalFrame&>(frame);
}

// Shared device state: queue, compiled kernels, texture cache.
class MetalContext {
 public:
  explicit MetalContext(id<MTLDevice> device) : device_(device) {
    queue_ = [device_ newCommandQueue];
    NSError* error = nil;
    library_ = [device_ newLibraryWithSource:@(kInterpolateShaderSource)
                                     options:[MTLCompileOptions new]
                                       error:&error];
    if (!library_) {
      throw std::runtime_error(std::string("Metal shader compile failed: ") +
                               error.localizedDescription.UTF8String);
    }
    if (CVMetalTextureCacheCreate(kCFAllocatorDefault, nullptr, device_, nullptr, &cache_) !=
        kCVReturnSuccess) {
      throw std::runtime_error("cannot create Metal texture cache");
    }
  }

  ~MetalContext() {
    for (auto& [key, pool] : pools_) CVPixelBufferPoolRelease(pool);
    if (cache_) CFRelease(cache_);
  }

  id<MTLDevice> device() const { return device_; }
  id<MTLCommandQueue> queue() const { return queue_; }
  CVMetalTextureCacheRef cache() const { return cache_; }

  id<MTLComputePipelineState> pipeline(const char* name) {
    auto it = pipelines_.find(name);
    if (it != pipelines_.end()) return it->second;
    id<MTLFunction> fn = [library_ newFunctionWithName:@(name)];
    if (!fn) throw std::runtime_error(std::string("missing Metal kernel ") + name);
    NSError* error = nil;
    id<MTLComputePipelineState> state = [device_ newComputePipelineStateWithFunction:fn
                                                                                error:&error];
    if (!state) throw std::runtime_error(std::string("cannot build Metal kernel ") + name);
    pipelines_[name] = state;
    return state;
  }

  id<MTLTexture> texture(int w, int h, MTLPixelFormat format) {
    MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                 width:w
                                                                                height:h
                                                                             mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    d.storageMode = MTLStorageModePrivate;
    return [device_ newTextureWithDescriptor:d];
  }

  GpuFramePtr new_frame(int w, int h, OSType format) {
    const auto key = std::make_tuple(w, h, format);
    auto it = pools_.find(key);
    if (it == pools_.end()) {
      NSDictionary* attrs = @{
        (id)kCVPixelBufferPixelFormatTypeKey : @(format),
        (id)kCVPixelBufferWidthKey : @(w),
        (id)kCVPixelBufferHeightKey : @(h),
        (id)kCVPixelBufferIOSurfacePropertiesKey : @{},
        (id)kCVPixelBufferMetalCompatibilityKey : @YES,
      };
      CVPixelBufferPoolRef pool = nullptr;
      if (CVPixelBufferPoolCreate(kCFAllocatorDefault, nullptr, (__bridge CFDictionaryRef)attrs,
                                  &pool) != kCVReturnSuccess) {
        throw std::runtime_error("cannot create pixel buffer pool");
      }
      it = pools_.emplace(key, pool).first;
    }
    CVPixelBufferRef buffer = nullptr;
    if (CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, it->second, &buffer) !=
        kCVReturnSuccess) {
      throw std::runtime_error("cannot allocate pixel buffer");
    }
    auto frame = std::make_shared<MetalFrame>(buffer, cache_);
    CVPixelBufferRelease(buffer);  // the frame holds its own reference
    return frame;
  }

  static void dispatch(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> state,
                       int w, int h) {
    [enc setComputePipelineState:state];
    [enc dispatchThreads:MTLSizeMake(w, h, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
  }

 private:
  id<MTLDevice> device_;
  id<MTLCommandQueue> queue_;
  id<MTLLibrary> library_;
  CVMetalTextureCacheRef cache_ = nullptr;
  std::map<std::string, id<MTLComputePipelineState>> pipelines_;
  std::map<std::tuple<int, int, OSType>, CVPixelBufferPoolRef> pools_;
};

// Shared parts of the Metal engines: the stats counter and the final
// interpolate_frame pass.
class MetalEngine : public Interpolator {
 public:
  MetalEngine(MetalContext& ctx, int mode, float cut_fraction)
      : ctx_(ctx), mode_(mode), cut_fraction_(cut_fraction) {
    stats_ = [ctx_.device() newBufferWithLength:16 options:MTLResourceStorageModeShared];
    dummy_field_ = ctx_.texture(1, 1, MTLPixelFormatRGBA32Float);
  }

  void interpolate(float t, GpuFrame& out) override {
    if (!a_ || !b_) throw std::logic_error("interpolate() before set_pair()");
    const auto& a = as_metal(*a_);
    const auto& b = as_metal(*b_);
    const auto& o = as_metal(out);

    InterpParams params{t,     mode_, cut_fraction_, total_blocks_, 0.06f, 0.16f,
                        bicubic_ ? 1 : 0, kCandidatePenalty, smooth_a() ? 0.5f : 1.0f};
    id<MTLCommandBuffer> cb = [ctx_.queue() commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setTexture:a.y() atIndex:0];
    [enc setTexture:a.uv() atIndex:1];
    [enc setTexture:b.y() atIndex:2];
    [enc setTexture:b.uv() atIndex:3];
    [enc setTexture:forward_field() atIndex:4];
    [enc setTexture:backward_field() atIndex:5];
    [enc setTexture:o.y() atIndex:6];
    [enc setTexture:o.uv() atIndex:7];
    [enc setTexture:(smooth_a() ?: a.y()) atIndex:8];
    [enc setTexture:(smooth_b() ?: b.y()) atIndex:9];
    [enc setBytes:&params length:sizeof params atIndex:0];
    [enc setBuffer:stats_ offset:0 atIndex:1];
    MetalContext::dispatch(enc, ctx_.pipeline("interpolate_frame"), (o.width() + 1) / 2,
                           (o.height() + 1) / 2);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) {
      throw std::runtime_error("GPU interpolation failed");
    }
    // Colour metadata (matrix, primaries, transfer) follows the source.
    CVBufferPropagateAttachments(a.buffer(), o.buffer());

    const uint32_t unmatched = *static_cast<const uint32_t*>(stats_.contents);
    stats_out_.unmatched = total_blocks_ ? static_cast<float>(unmatched) / total_blocks_ : 0.f;
    stats_out_.scene_cut = stats_out_.unmatched > cut_fraction_;
  }

  PairStats last_stats() const override { return stats_out_; }

 protected:
  virtual id<MTLTexture> forward_field() const { return dummy_field_; }
  virtual id<MTLTexture> backward_field() const { return dummy_field_; }
  // Half-resolution luma of a and b, for match scoring; nil if none.
  virtual id<MTLTexture> smooth_a() const { return nil; }
  virtual id<MTLTexture> smooth_b() const { return nil; }

  void check_pair(const GpuFramePtr& a, const GpuFramePtr& b) const {
    if (!a || !b || a->width() != b->width() || a->height() != b->height()) {
      throw std::invalid_argument("source frames must have the same size");
    }
  }

  void reset_stats(id<MTLCommandBuffer> cb) {
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit fillBuffer:stats_ range:NSMakeRange(0, 16) value:0];
    [blit endEncoding];
  }

  MetalContext& ctx_;
  int mode_;
  float cut_fraction_;
  id<MTLBuffer> stats_;
  id<MTLTexture> dummy_field_;
  GpuFramePtr a_;
  GpuFramePtr b_;
  uint32_t total_blocks_ = 0;
  bool bicubic_ = false;
  PairStats stats_out_;
};

// L1: cross-fade between the two source frames.
class BlendEngine final : public MetalEngine {
 public:
  explicit BlendEngine(MetalContext& ctx) : MetalEngine(ctx, 1, 0.7f) {}

  std::string name() const override { return "blend"; }
  Tier tier() const override { return Tier::Blend; }

  void set_pair(const GpuFramePtr& a, const GpuFramePtr& b) override {
    check_pair(a, b);
    a_ = a;
    b_ = b;
    const int bw = blocks(a->width());
    const int bh = blocks(a->height());
    total_blocks_ = static_cast<uint32_t>(bw * bh);

    SearchParams params{};
    params.field_size = simd_make_int2(bw, bh);
    params.bad_sad = 24.f;
    id<MTLCommandBuffer> cb = [ctx_.queue() commandBuffer];
    reset_stats(cb);
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setTexture:as_metal(*a).y() atIndex:0];
    [enc setTexture:as_metal(*b).y() atIndex:1];
    [enc setBytes:&params length:sizeof params atIndex:0];
    [enc setBuffer:stats_ offset:0 atIndex:1];
    MetalContext::dispatch(enc, ctx_.pipeline("block_difference"), bw, bh);
    [enc endEncoding];
    [cb commit];
  }
};

struct MemcSettings {
  const char* name;
  int quality_rank;
  int radius;        // coarsest-level search radius
  int refine_iters;  // per level
  int subpel;        // finest level: 0 integer, 1 half-pel, 2 quarter-pel
  float lambda;
  // Relative SAD margin a non-zero vector needs over zero motion. Keeps
  // unmatched noise (grain, rain) from picking up chance or inherited motion.
  float new_penalty;
  bool bicubic;      // sharper output sampling
};

// L2: hierarchical block matching + bidirectional motion compensation.
class MemcEngine final : public MetalEngine {
 public:
  MemcEngine(MetalContext& ctx, MemcSettings settings)
      : MetalEngine(ctx, 0, 0.55f), settings_(settings) {
    bicubic_ = settings.bicubic;
  }

  std::string name() const override { return settings_.name; }
  Tier tier() const override { return Tier::Memc; }

  void set_pair(const GpuFramePtr& a, const GpuFramePtr& b) override {
    check_pair(a, b);
    allocate(a->width(), a->height());

    id<MTLCommandBuffer> cb = [ctx_.queue() commandBuffer];
    reset_stats(cb);
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    // Sliding pairs (a,b) then (b,c): b's pyramid is already built.
    if (b_ && a.get() == b_.get()) {
      std::swap(pyramid_a_, pyramid_b_);
    } else {
      build_pyramid(enc, *a, pyramid_a_);
    }
    build_pyramid(enc, *b, pyramid_b_);
    a_ = a;
    b_ = b;

    const int levels = static_cast<int>(levels_.size());
    for (int level = levels - 1; level >= 0; --level) {
      const bool coarsest = level == levels - 1;
      search(enc, level, pyramid_a_, pyramid_b_, coarsest ? nil : fwd_[level + 1], fwd_[level],
             level == 0);
      search(enc, level, pyramid_b_, pyramid_a_, coarsest ? nil : bwd_[level + 1], bwd_[level],
             false);
    }
    median(enc, fwd_[0], fwd_filtered_);
    median(enc, bwd_[0], bwd_filtered_);
    [enc endEncoding];
    [cb commit];  // interpolate() runs after this on the same queue
  }

 protected:
  id<MTLTexture> forward_field() const override { return fwd_filtered_; }
  id<MTLTexture> backward_field() const override { return bwd_filtered_; }
  id<MTLTexture> smooth_a() const override { return levels_.size() > 1 ? pyramid_a_.at(1) : nil; }
  id<MTLTexture> smooth_b() const override { return levels_.size() > 1 ? pyramid_b_.at(1) : nil; }

 private:
  struct Level {
    int width;
    int height;
  };
  // Level 0 is the frame's own luma plane; deeper levels are owned textures.
  struct Pyramid {
    id<MTLTexture> level0;
    std::vector<id<MTLTexture>> owned;
    id<MTLTexture> at(int level) const { return level == 0 ? level0 : owned[level - 1]; }
  };

  void allocate(int w, int h) {
    if (w == width_ && h == height_) return;
    width_ = w;
    height_ = h;
    levels_.clear();
    // Coarsest level stays around 160+ pixels wide: enough detail to lock on,
    // small enough that an exhaustive search covers large motion.
    for (int lw = w, lh = h; levels_.empty() || (lw >= 160 && levels_.size() < 6);
         lw = (lw + 1) / 2, lh = (lh + 1) / 2) {
      levels_.push_back({lw, lh});
    }
    for (Pyramid* p : {&pyramid_a_, &pyramid_b_}) {
      p->owned.clear();
      for (size_t i = 1; i < levels_.size(); ++i) {
        p->owned.push_back(ctx_.texture(levels_[i].width, levels_[i].height, MTLPixelFormatR16Float));
      }
    }
    fwd_.clear();
    bwd_.clear();
    for (const auto& l : levels_) {
      fwd_.push_back(ctx_.texture(blocks(l.width), blocks(l.height), MTLPixelFormatRGBA32Float));
      bwd_.push_back(ctx_.texture(blocks(l.width), blocks(l.height), MTLPixelFormatRGBA32Float));
    }
    fwd_filtered_ = ctx_.texture(blocks(w), blocks(h), MTLPixelFormatRGBA32Float);
    bwd_filtered_ = ctx_.texture(blocks(w), blocks(h), MTLPixelFormatRGBA32Float);
    total_blocks_ = static_cast<uint32_t>(blocks(w) * blocks(h));
    b_.reset();  // cached pyramids no longer match
  }

  void build_pyramid(id<MTLComputeCommandEncoder> enc, const GpuFrame& frame, Pyramid& p) {
    p.level0 = as_metal(frame).y();
    for (size_t i = 1; i < levels_.size(); ++i) {
      [enc setTexture:p.at(static_cast<int>(i) - 1) atIndex:0];
      [enc setTexture:p.at(static_cast<int>(i)) atIndex:1];
      MetalContext::dispatch(enc, ctx_.pipeline("downsample2x"), levels_[i].width,
                             levels_[i].height);
    }
  }

  void search(id<MTLComputeCommandEncoder> enc, int level, const Pyramid& cur, const Pyramid& ref,
              id<MTLTexture> coarse, id<MTLTexture> out, bool count_stats) {
    const Level& l = levels_[level];
    SearchParams params{};
    params.field_size = simd_make_int2(blocks(l.width), blocks(l.height));
    params.coarse_field_size = coarse ? simd_make_int2(static_cast<int>(coarse.width),
                                                       static_cast<int>(coarse.height))
                                      : simd_make_int2(0, 0);
    params.radius = settings_.radius;
    params.lambda = settings_.lambda;
    params.refine_iters = settings_.refine_iters;
    params.subpel = level == 0 ? settings_.subpel : 0;
    params.bad_sad = 20.f;
    params.count_stats = count_stats ? 1 : 0;
    params.new_penalty = settings_.new_penalty;

    [enc setTexture:cur.at(level) atIndex:0];
    [enc setTexture:ref.at(level) atIndex:1];
    [enc setTexture:(coarse ? coarse : dummy_field_) atIndex:2];
    [enc setTexture:out atIndex:3];
    [enc setBytes:&params length:sizeof params atIndex:0];
    [enc setBuffer:stats_ offset:0 atIndex:1];
    MetalContext::dispatch(enc, ctx_.pipeline("motion_search"), params.field_size.x,
                           params.field_size.y);
  }

  void median(id<MTLComputeCommandEncoder> enc, id<MTLTexture> in, id<MTLTexture> out) {
    [enc setTexture:in atIndex:0];
    [enc setTexture:out atIndex:1];
    MetalContext::dispatch(enc, ctx_.pipeline("vector_median"), static_cast<int>(in.width),
                           static_cast<int>(in.height));
  }

  MemcSettings settings_;
  int width_ = 0;
  int height_ = 0;
  std::vector<Level> levels_;
  Pyramid pyramid_a_;
  Pyramid pyramid_b_;
  std::vector<id<MTLTexture>> fwd_;
  std::vector<id<MTLTexture>> bwd_;
  id<MTLTexture> fwd_filtered_;
  id<MTLTexture> bwd_filtered_;
};

// Measured with tests/quality.sh: quarter-pel (subpel 2) gains <0.1 dB; half-pel
// on motion-fast gains 0.1 dB for 14% more cost; bicubic on motion-fast gains
// 1.2 dB but doubles its per-frame cost, which 4K real time cannot spare.
// new_penalty 0.05: 0.1 cost pans 0.15 dB for no gain elsewhere; 0 leaves
// changing noise (grain) free to pick up chance motion.
constexpr MemcSettings kMemcQuality{"motion", 20, 8, 2, 1, 0.35f, 0.05f, true};
constexpr MemcSettings kMemcFast{"motion-fast", 10, 6, 1, 0, 0.35f, 0.05f, false};

class MetalBackend final : public GpuBackend {
 public:
  explicit MetalBackend(id<MTLDevice> device) : ctx_(device) {}

  std::string name() const override {
    return std::string("Metal (") + ctx_.device().name.UTF8String + ")";
  }

  std::vector<EngineInfo> engines() const override {
    return {
        {"blend", Tier::Blend, 0, "cross-fade between source frames"},
        {kMemcFast.name, Tier::Memc, kMemcFast.quality_rank,
         "motion compensation, faster search"},
        {kMemcQuality.name, Tier::Memc, kMemcQuality.quality_rank,
         "motion compensation, half-pixel search, bicubic output"},
    };
  }

  std::unique_ptr<Interpolator> create_engine(const std::string& name) override {
    if (name == "blend") return std::make_unique<BlendEngine>(ctx_);
    if (name == kMemcFast.name) return std::make_unique<MemcEngine>(ctx_, kMemcFast);
    if (name == kMemcQuality.name) return std::make_unique<MemcEngine>(ctx_, kMemcQuality);
    throw std::invalid_argument("unknown engine: " + name);
  }

  GpuFramePtr wrap_native(void* handle) override {
    return std::make_shared<MetalFrame>(static_cast<CVPixelBufferRef>(handle), ctx_.cache());
  }

  GpuFramePtr new_frame(int width, int height, bool high_bit_depth) override {
    return ctx_.new_frame(width, height,
                          high_bit_depth ? kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
                                         : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange);
  }

  GpuFramePtr new_frame_like(const GpuFrame& like) override {
    return ctx_.new_frame(like.width(), like.height(),
                          CVPixelBufferGetPixelFormatType(as_metal(like).buffer()));
  }

  void write_planes(GpuFrame& frame, const uint8_t* y, int y_stride, const uint8_t* uv,
                    int uv_stride) override {
    CVPixelBufferRef buffer = as_metal(frame).buffer();
    CVPixelBufferLockBaseAddress(buffer, 0);
    copy_plane(buffer, 0, y, y_stride, true, frame.high_bit_depth());
    copy_plane(buffer, 1, uv, uv_stride, true, frame.high_bit_depth());
    CVPixelBufferUnlockBaseAddress(buffer, 0);
  }

  void read_planes(const GpuFrame& frame, uint8_t* y, int y_stride, uint8_t* uv,
                   int uv_stride) override {
    CVPixelBufferRef buffer = as_metal(frame).buffer();
    CVPixelBufferLockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
    copy_plane(buffer, 0, y, y_stride, false, frame.high_bit_depth());
    copy_plane(buffer, 1, uv, uv_stride, false, frame.high_bit_depth());
    CVPixelBufferUnlockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
  }

 private:
  // Copies one plane between CPU memory and the pixel buffer, row by row
  // because the two sides use different strides.
  static void copy_plane(CVPixelBufferRef buffer, size_t plane, const uint8_t* cpu_in,
                         int cpu_stride, bool to_buffer, bool high) {
    auto* base = static_cast<uint8_t*>(CVPixelBufferGetBaseAddressOfPlane(buffer, plane));
    const size_t stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, plane);
    const size_t rows = CVPixelBufferGetHeightOfPlane(buffer, plane);
    // Plane 0: one sample per pixel; plane 1: interleaved Cb/Cr pairs.
    const size_t samples = CVPixelBufferGetWidthOfPlane(buffer, plane) * (plane == 0 ? 1 : 2);
    const size_t row_bytes = samples * (high ? 2 : 1);
    auto* cpu = const_cast<uint8_t*>(cpu_in);
    for (size_t r = 0; r < rows; ++r) {
      uint8_t* gpu_row = base + r * stride;
      uint8_t* cpu_row = cpu + r * static_cast<size_t>(cpu_stride);
      if (to_buffer) {
        std::memcpy(gpu_row, cpu_row, row_bytes);
      } else {
        std::memcpy(cpu_row, gpu_row, row_bytes);
      }
    }
  }

  MetalContext ctx_;
};

}  // namespace

std::unique_ptr<GpuBackend> GpuBackend::create_default() {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) return nullptr;
  return std::make_unique<MetalBackend>(device);
}

}  // namespace advp
