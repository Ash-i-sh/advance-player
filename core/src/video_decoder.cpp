#include "video_decoder.h"

#include <cmath>
#include <stdexcept>

#include "ffmpeg_util.h"

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

namespace advp {
namespace {

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

}  // namespace

VideoDecoder::VideoDecoder(GpuBackend& backend, const std::string& path) : backend_(backend) {
  packet_ = av_packet_alloc();
  frame_ = av_frame_alloc();
  props_ = av_frame_alloc();
  try {
    check(avformat_open_input(&format_, path.c_str(), nullptr, nullptr), "open input");
    check(avformat_find_stream_info(format_, nullptr), "read stream info");
    video_index_ = av_find_best_stream(format_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    check(video_index_, "find video stream");
    video_ = format_->streams[video_index_];

    AVRational fps = video_->avg_frame_rate;
    if (fps.num <= 0 || fps.den <= 0) fps = video_->r_frame_rate;
    if (fps.num <= 0 || fps.den <= 0) fps = AVRational{24, 1};
    frame_rate_ = Rational{fps.num, fps.den};
    if (format_->duration > 0) duration_ = format_->duration / static_cast<double>(AV_TIME_BASE);

    const AVCodec* codec = avcodec_find_decoder(video_->codecpar->codec_id);
    if (!codec) throw std::runtime_error("no decoder for this video codec");
    codec_ = avcodec_alloc_context3(codec);
    check(avcodec_parameters_to_context(codec_, video_->codecpar), "decoder parameters");
    codec_->pkt_timebase = video_->time_base;
    AVBufferRef* hw = nullptr;
    if (av_hwdevice_ctx_create(&hw, kHwDevice, nullptr, nullptr, 0) >= 0) {
      codec_->hw_device_ctx = hw;  // the context takes our reference
      codec_->get_format = pick_format;
    }
    check(avcodec_open2(codec_, codec, nullptr), "open decoder");
  } catch (...) {
    release();
    throw;
  }
}

VideoDecoder::~VideoDecoder() { release(); }

void VideoDecoder::release() {
  avcodec_free_context(&codec_);
  avformat_close_input(&format_);
  av_packet_free(&packet_);
  av_frame_free(&frame_);
  av_frame_free(&props_);
  av_frame_free(&sw_frame_);
  sws_freeContext(sws_);
  sws_ = nullptr;
}

bool VideoDecoder::next(DecodedFrame& out, const std::function<void(AVPacket*)>& on_other) {
  while (true) {
    const int rc = avcodec_receive_frame(codec_, frame_);
    if (rc == AVERROR_EOF) return false;
    if (rc != AVERROR(EAGAIN)) {
      check(rc, "decode frame");
      out.frame = to_gpu(frame_);
      av_frame_unref(props_);
      av_frame_copy_props(props_, frame_);
      props_->width = frame_->width;
      props_->height = frame_->height;

      int64_t ts = frame_->best_effort_timestamp;
      if (ts == AV_NOPTS_VALUE) ts = frame_->pts;
      const double step = 1.0 / frame_rate_.value();
      out.time = ts == AV_NOPTS_VALUE ? (have_time_ ? last_time_ + step : 0.0)
                                      : ts * av_q2d(video_->time_base);
      last_time_ = out.time;
      have_time_ = true;
      av_frame_unref(frame_);
      return true;
    }
    if (draining_) return false;

    // The decoder wants input: feed it the next video packet.
    while (true) {
      const int read = av_read_frame(format_, packet_);
      if (read < 0) {
        avcodec_send_packet(codec_, nullptr);
        draining_ = true;
        break;
      }
      if (packet_->stream_index == video_index_) {
        const int sent = avcodec_send_packet(codec_, packet_);
        av_packet_unref(packet_);
        check(sent, "decode");
        break;
      }
      if (on_other) on_other(packet_);
      av_packet_unref(packet_);
    }
  }
}

void VideoDecoder::seek(double seconds) {
  const int64_t ts = static_cast<int64_t>(std::llround(seconds / av_q2d(video_->time_base)));
  check(av_seek_frame(format_, video_index_, ts, AVSEEK_FLAG_BACKWARD), "seek");
  avcodec_flush_buffers(codec_);
  draining_ = false;
  have_time_ = false;
}

GpuFramePtr VideoDecoder::to_gpu(AVFrame* frame) {
  if (frame->format == kHwFormat) {
    hardware_ = true;
    return backend_.wrap_native(frame->data[3]);
  }
  hardware_ = false;
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

}  // namespace advp
