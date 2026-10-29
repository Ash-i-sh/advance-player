#include "advp/convert.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "advp/rate.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

namespace advp {
namespace {

std::string av_error(int code) {
  char buf[AV_ERROR_MAX_STRING_SIZE] = {};
  av_strerror(code, buf, sizeof buf);
  return buf;
}

void check(int code, const char* what) {
  if (code < 0) throw std::runtime_error(std::string(what) + ": " + av_error(code));
}

struct FormatInput {
  AVFormatContext* ctx = nullptr;
  ~FormatInput() { avformat_close_input(&ctx); }
};
struct FormatOutput {
  AVFormatContext* ctx = nullptr;
  bool opened = false;
  ~FormatOutput() {
    if (ctx && opened && !(ctx->oformat->flags & AVFMT_NOFILE)) avio_closep(&ctx->pb);
    avformat_free_context(ctx);
  }
};
struct CodecContext {
  AVCodecContext* ctx = nullptr;
  ~CodecContext() { avcodec_free_context(&ctx); }
};
struct PacketPtr {
  AVPacket* pkt = av_packet_alloc();
  ~PacketPtr() { av_packet_free(&pkt); }
};
struct FramePtr {
  AVFrame* frame = av_frame_alloc();
  ~FramePtr() { av_frame_free(&frame); }
};

#if defined(__APPLE__)
constexpr AVHWDeviceType kHwDevice = AV_HWDEVICE_TYPE_VIDEOTOOLBOX;
constexpr AVPixelFormat kHwFormat = AV_PIX_FMT_VIDEOTOOLBOX;
#endif

AVPixelFormat pick_format(AVCodecContext*, const AVPixelFormat* formats) {
  for (const AVPixelFormat* f = formats; *f != AV_PIX_FMT_NONE; ++f) {
    if (*f == kHwFormat) return *f;
  }
  // No hardware path for this stream: take the first software format.
  for (const AVPixelFormat* f = formats; *f != AV_PIX_FMT_NONE; ++f) {
    const AVPixFmtDescriptor* d = av_pix_fmt_desc_get(*f);
    if (d && !(d->flags & AV_PIX_FMT_FLAG_HWACCEL)) return *f;
  }
  return formats[0];
}

// Keeps a GpuFrame alive while the encoder holds the AVFrame that wraps it.
void release_gpu_frame(void* opaque, uint8_t*) { delete static_cast<GpuFramePtr*>(opaque); }

bool is_mp4_family(const AVOutputFormat* f) {
  const std::string name = f->name;
  return name.find("mp4") != std::string::npos || name.find("mov") != std::string::npos;
}

class Converter {
 public:
  Converter(GpuBackend& backend, const ConvertOptions& options)
      : backend_(backend), opt_(options) {}

  ConvertResult run() {
    start_clock_ = std::chrono::steady_clock::now();
    open_input();
    open_decoder();
    open_output();
    pick_engine();
    process();
    result_.ok = true;
    result_.seconds = elapsed();
    return result_;
  }

 private:
  double elapsed() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_clock_).count();
  }

  void open_input() {
    check(avformat_open_input(&in_.ctx, opt_.input.c_str(), nullptr, nullptr), "open input");
    check(avformat_find_stream_info(in_.ctx, nullptr), "read stream info");
    video_index_ = av_find_best_stream(in_.ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    check(video_index_, "find video stream");
    video_ = in_.ctx->streams[video_index_];

    AVRational fps = video_->avg_frame_rate;
    if (fps.num <= 0 || fps.den <= 0) fps = video_->r_frame_rate;
    if (fps.num <= 0 || fps.den <= 0) fps = AVRational{24, 1};
    source_fps_ = Rational{fps.num, fps.den};
    const auto target = resolve_target_fps(opt_.target_fps.c_str(), source_fps_);
    if (!target) throw std::runtime_error("invalid target frame rate: " + opt_.target_fps);
    target_fps_ = *target;
    rate_ = RateConverter(target_fps_);
    result_.source_fps = source_fps_.value();
    result_.target_fps = target_fps_.value();
    if (in_.ctx->duration > 0) duration_ = in_.ctx->duration / static_cast<double>(AV_TIME_BASE);
  }

  void open_decoder() {
    const AVCodec* codec = avcodec_find_decoder(video_->codecpar->codec_id);
    if (!codec) throw std::runtime_error("no decoder for this video codec");
    dec_.ctx = avcodec_alloc_context3(codec);
    check(avcodec_parameters_to_context(dec_.ctx, video_->codecpar), "decoder parameters");
    dec_.ctx->pkt_timebase = video_->time_base;
    AVBufferRef* hw = nullptr;
    if (av_hwdevice_ctx_create(&hw, kHwDevice, nullptr, nullptr, 0) >= 0) {
      dec_.ctx->hw_device_ctx = hw;  // the context takes our reference
      dec_.ctx->get_format = pick_format;
    }
    check(avcodec_open2(dec_.ctx, codec, nullptr), "open decoder");
  }

  void open_output() {
    check(avformat_alloc_output_context2(&out_.ctx, nullptr, nullptr, opt_.output.c_str()),
          "create output");
    const bool mp4 = is_mp4_family(out_.ctx->oformat);

    stream_map_.assign(in_.ctx->nb_streams, -1);
    for (unsigned i = 0; i < in_.ctx->nb_streams; ++i) {
      AVStream* in = in_.ctx->streams[i];
      if (static_cast<int>(i) == video_index_) {
        AVStream* out = avformat_new_stream(out_.ctx, nullptr);
        stream_map_[i] = out->index;
        continue;
      }
      const AVMediaType type = in->codecpar->codec_type;
      const bool keep = type == AVMEDIA_TYPE_AUDIO ||
                        (type == AVMEDIA_TYPE_SUBTITLE &&
                         (!mp4 || in->codecpar->codec_id == AV_CODEC_ID_MOV_TEXT));
      if (!keep) continue;
      AVStream* out = avformat_new_stream(out_.ctx, nullptr);
      check(avcodec_parameters_copy(out->codecpar, in->codecpar), "copy stream parameters");
      out->codecpar->codec_tag = 0;
      out->time_base = in->time_base;
      av_dict_copy(&out->metadata, in->metadata, 0);
      out->disposition = in->disposition;
      stream_map_[i] = out->index;
    }
    if (!(out_.ctx->oformat->flags & AVFMT_NOFILE)) {
      check(avio_open(&out_.ctx->pb, opt_.output.c_str(), AVIO_FLAG_WRITE), "open output file");
      out_.opened = true;
    }
  }

  // The encoder needs the decoded pixel format, so it opens on the first frame.
  void open_encoder(const AVFrame* first, bool high) {
    std::string codec_name = opt_.codec == "h264" ? "h264_videotoolbox" : "hevc_videotoolbox";
    if (high) codec_name = "hevc_videotoolbox";  // H.264 hardware encode is 8-bit only
    const AVCodec* codec = avcodec_find_encoder_by_name(codec_name.c_str());
    if (!codec) throw std::runtime_error("encoder not available: " + codec_name);

    enc_.ctx = avcodec_alloc_context3(codec);
    AVCodecContext* e = enc_.ctx;
    e->width = first->width;
    e->height = first->height;
    e->time_base = AVRational{static_cast<int>(target_fps_.den), static_cast<int>(target_fps_.num)};
    e->framerate = AVRational{static_cast<int>(target_fps_.num), static_cast<int>(target_fps_.den)};
    e->pix_fmt = kHwFormat;
    e->sw_pix_fmt = high ? AV_PIX_FMT_P010LE : AV_PIX_FMT_NV12;
    e->sample_aspect_ratio = first->sample_aspect_ratio;
    e->color_range = first->color_range;
    e->color_primaries = first->color_primaries;
    e->color_trc = first->color_trc;
    e->colorspace = first->colorspace;
    e->chroma_sample_location = first->chroma_location;
    e->flags |= AV_CODEC_FLAG_QSCALE;
    e->global_quality = std::clamp(opt_.quality, 1, 100) * FF_QP2LAMBDA;
    if (high) av_opt_set(e->priv_data, "profile", "main10", 0);
    if (out_.ctx->oformat->flags & AVFMT_GLOBALHEADER) e->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    check(avcodec_open2(e, codec, nullptr), "open hardware encoder");

    AVStream* out = out_.ctx->streams[stream_map_[video_index_]];
    check(avcodec_parameters_from_context(out->codecpar, e), "encoder parameters");
    if (codec->id == AV_CODEC_ID_HEVC && is_mp4_family(out_.ctx->oformat)) {
      out->codecpar->codec_tag = MKTAG('h', 'v', 'c', '1');  // plays in QuickTime/Safari
    }
    out->time_base = e->time_base;
    out->avg_frame_rate = e->framerate;
    av_dict_copy(&out->metadata, video_->metadata, 0);

    check(avformat_write_header(out_.ctx, nullptr), "write header");
    header_written_ = true;
    for (AVPacket* pending : pending_packets_) write_copied(pending);
    pending_packets_.clear();
  }

  void pick_engine() {
    std::string name = opt_.engine;
    if (name == "auto") {
      // Offline export has no real-time limit: take the best-looking engine.
      const auto engines = backend_.engines();
      const auto best = std::max_element(engines.begin(), engines.end(), [](auto& a, auto& b) {
        return std::tie(a.tier, a.quality_rank) < std::tie(b.tier, b.quality_rank);
      });
      if (best == engines.end()) throw std::runtime_error("no interpolation engine available");
      name = best->name;
    }
    engine_ = backend_.create_engine(name);
    result_.engine = engine_->name();
  }

  void process() {
    PacketPtr packet;
    FramePtr frame;
    bool stop = false;
    while (!stop && av_read_frame(in_.ctx, packet.pkt) >= 0) {
      const int index = packet.pkt->stream_index;
      if (index == video_index_) {
        check(avcodec_send_packet(dec_.ctx, packet.pkt), "decode");
        stop = drain_decoder(frame.frame);
      } else if (index < static_cast<int>(stream_map_.size()) && stream_map_[index] >= 0) {
        AVPacket* copy = av_packet_clone(packet.pkt);
        if (header_written_) {
          write_copied(copy);
        } else {
          pending_packets_.push_back(copy);
        }
      }
      av_packet_unref(packet.pkt);
    }
    if (!stop) {
      avcodec_send_packet(dec_.ctx, nullptr);
      drain_decoder(frame.frame);
    }
    if (!header_written_) throw std::runtime_error("no video frames decoded");

    // Hold the last frame for one source frame duration.
    emit(rate_.flush(last_time_ + 1.0 / source_fps_.value()));
    check(avcodec_send_frame(enc_.ctx, nullptr), "flush encoder");
    drain_encoder();
    check(av_write_trailer(out_.ctx), "write trailer");
    for (AVPacket* p : pending_packets_) av_packet_free(&p);
    pending_packets_.clear();
  }

  // Returns true once the frame limit is reached.
  bool drain_decoder(AVFrame* frame) {
    while (true) {
      const int rc = avcodec_receive_frame(dec_.ctx, frame);
      if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) return false;
      check(rc, "decode frame");
      on_decoded(frame);
      av_frame_unref(frame);
      if (opt_.max_source_frames > 0 && result_.source_frames >= opt_.max_source_frames) {
        return true;
      }
    }
  }

  void on_decoded(AVFrame* frame) {
    GpuFramePtr gpu;
    if (frame->format == kHwFormat) {
      gpu = backend_.wrap_native(frame->data[3]);
      result_.decoder = "hardware";
    } else {
      gpu = upload_software(frame);
      result_.decoder = "software";
    }
    if (!enc_.ctx) {
      color_template_ = av_frame_alloc();
      av_frame_copy_props(color_template_, frame);
      color_template_->width = frame->width;
      color_template_->height = frame->height;
      open_encoder(frame, gpu->high_bit_depth());
    }

    int64_t ts = frame->best_effort_timestamp;
    if (ts == AV_NOPTS_VALUE) ts = frame->pts;
    double time = ts == AV_NOPTS_VALUE ? last_time_ + 1.0 / source_fps_.value()
                                       : ts * av_q2d(video_->time_base);
    ++result_.source_frames;

    cur_ = gpu;
    const auto slots = rate_.push(time);
    emit(slots);
    prev_ = cur_;
    pair_ready_ = false;
    last_time_ = time;

    if (opt_.on_progress && result_.source_frames % 24 == 0) {
      ConvertProgress p;
      p.source_frames = result_.source_frames;
      p.output_frames = result_.output_frames;
      p.media_seconds = time - rate_.start_time();
      p.duration = duration_;
      p.elapsed = elapsed();
      opt_.on_progress(p);
    }
  }

  GpuFramePtr upload_software(const AVFrame* frame) {
    const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(frame->format));
    const bool high = desc && desc->comp[0].depth > 8;
    const AVPixelFormat target = high ? AV_PIX_FMT_P010LE : AV_PIX_FMT_NV12;
    sws_ = sws_getCachedContext(sws_, frame->width, frame->height,
                                static_cast<AVPixelFormat>(frame->format), frame->width,
                                frame->height, target, SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws_) throw std::runtime_error("cannot convert software-decoded frame");
    if (!sw_frame_ || sw_frame_->width != frame->width || sw_frame_->height != frame->height ||
        sw_frame_->format != target) {
      av_frame_free(&sw_frame_);
      sw_frame_ = av_frame_alloc();
      sw_frame_->format = target;
      sw_frame_->width = frame->width;
      sw_frame_->height = frame->height;
      check(av_frame_get_buffer(sw_frame_, 0), "allocate conversion frame");
    }
    sws_scale(sws_, frame->data, frame->linesize, 0, frame->height, sw_frame_->data,
              sw_frame_->linesize);
    auto gpu = backend_.new_frame(frame->width, frame->height, high);
    backend_.write_planes(*gpu, sw_frame_->data[0], sw_frame_->linesize[0], sw_frame_->data[1],
                          sw_frame_->linesize[1]);
    return gpu;
  }

  void emit(const std::vector<OutputSlot>& slots) {
    for (const auto& slot : slots) {
      GpuFramePtr out;
      if (!prev_ || slot.t <= 0.f) {
        out = prev_ ? prev_ : cur_;
      } else if (slot.t >= 1.f) {
        out = cur_;
      } else {
        if (!pair_ready_) {
          engine_->set_pair(prev_, cur_);
          pair_ready_ = true;
          pair_counted_ = false;
        }
        out = backend_.new_frame_like(*prev_);
        engine_->interpolate(slot.t, *out);
        ++result_.interpolated;
        if (!pair_counted_ && engine_->last_stats().scene_cut) {
          ++result_.scene_cuts;
          pair_counted_ = true;
        }
      }
      encode(out, slot);
    }
  }

  void encode(const GpuFramePtr& gpu, const OutputSlot& slot) {
    FramePtr f;
    check(av_frame_copy_props(f.frame, color_template_), "frame properties");
    f.frame->format = kHwFormat;
    f.frame->width = gpu->width();
    f.frame->height = gpu->height();
    f.frame->data[3] = static_cast<uint8_t*>(gpu->native_handle());
    f.frame->buf[0] = av_buffer_create(f.frame->data[3], 0, release_gpu_frame,
                                       new GpuFramePtr(gpu), AV_BUFFER_FLAG_READONLY);
    if (!f.frame->buf[0]) throw std::runtime_error("out of memory");
    f.frame->pts = std::llround(slot.time * target_fps_.num / static_cast<double>(target_fps_.den));
    f.frame->duration = 1;
    check(avcodec_send_frame(enc_.ctx, f.frame), "encode frame");
    ++result_.output_frames;
    drain_encoder();
  }

  void drain_encoder() {
    PacketPtr packet;
    AVStream* out = out_.ctx->streams[stream_map_[video_index_]];
    while (true) {
      const int rc = avcodec_receive_packet(enc_.ctx, packet.pkt);
      if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) return;
      check(rc, "encode");
      packet.pkt->stream_index = out->index;
      av_packet_rescale_ts(packet.pkt, enc_.ctx->time_base, out->time_base);
      check(av_interleaved_write_frame(out_.ctx, packet.pkt), "write video");
    }
  }

  void write_copied(AVPacket* packet) {
    AVStream* in = in_.ctx->streams[packet->stream_index];
    AVStream* out = out_.ctx->streams[stream_map_[packet->stream_index]];
    packet->stream_index = out->index;
    av_packet_rescale_ts(packet, in->time_base, out->time_base);
    packet->pos = -1;
    const int rc = av_interleaved_write_frame(out_.ctx, packet);
    av_packet_free(&packet);
    check(rc, "write copied stream");
  }

 public:
  ~Converter() {
    for (AVPacket* p : pending_packets_) av_packet_free(&p);
    av_frame_free(&color_template_);
    av_frame_free(&sw_frame_);
    sws_freeContext(sws_);
  }

 private:
  GpuBackend& backend_;
  ConvertOptions opt_;
  ConvertResult result_;
  std::chrono::steady_clock::time_point start_clock_;

  FormatInput in_;
  FormatOutput out_;
  CodecContext dec_;
  CodecContext enc_;
  int video_index_ = -1;
  AVStream* video_ = nullptr;
  std::vector<int> stream_map_;
  std::vector<AVPacket*> pending_packets_;
  bool header_written_ = false;
  AVFrame* color_template_ = nullptr;
  AVFrame* sw_frame_ = nullptr;
  SwsContext* sws_ = nullptr;

  Rational source_fps_;
  Rational target_fps_;
  double duration_ = 0;
  RateConverter rate_{Rational{60, 1}};
  std::unique_ptr<Interpolator> engine_;
  GpuFramePtr prev_;
  GpuFramePtr cur_;
  bool pair_ready_ = false;
  bool pair_counted_ = false;
  double last_time_ = 0;
};

}  // namespace

ConvertResult convert_file(GpuBackend& backend, const ConvertOptions& options) {
  try {
    Converter converter(backend, options);
    return converter.run();
  } catch (const std::exception& e) {
    ConvertResult r;
    r.error = e.what();
    return r;
  }
}

}  // namespace advp
