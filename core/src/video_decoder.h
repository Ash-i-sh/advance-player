// Opens a media file and decodes its main video stream into GPU frames,
// using the hardware decoder when the stream allows it. Shared by the
// converter and the player.
#pragma once

#include <functional>
#include <string>

#include "advp/engine.h"
#include "advp/rate.h"

struct AVFormatContext;
struct AVStream;
struct AVFrame;
struct AVPacket;
struct AVCodecContext;
struct SwsContext;

namespace advp {

struct DecodedFrame {
  GpuFramePtr frame;
  double time = 0;  // presentation time in seconds
};

class VideoDecoder {
 public:
  VideoDecoder(GpuBackend& backend, const std::string& path);
  ~VideoDecoder();
  VideoDecoder(const VideoDecoder&) = delete;
  VideoDecoder& operator=(const VideoDecoder&) = delete;

  AVFormatContext* format() const { return format_; }
  int video_index() const { return video_index_; }
  AVStream* video_stream() const { return video_; }
  Rational frame_rate() const { return frame_rate_; }
  double duration() const { return duration_; }  // seconds, 0 if unknown
  // True while frames come from the hardware decoder.
  bool hardware() const { return hardware_; }
  // Size, colour and aspect properties of the most recent frame.
  const AVFrame* properties() const { return props_; }

  // Decodes up to the next video frame. Packets of other streams go to
  // on_other; the decoder keeps ownership, so copy what you keep. Returns
  // false at the end of the stream.
  bool next(DecodedFrame& out, const std::function<void(AVPacket*)>& on_other = {});

  // Jumps to the key frame at or before seconds.
  void seek(double seconds);

 private:
  void release();
  GpuFramePtr to_gpu(AVFrame* frame);

  GpuBackend& backend_;
  AVFormatContext* format_ = nullptr;
  AVCodecContext* codec_ = nullptr;
  AVStream* video_ = nullptr;
  int video_index_ = -1;
  Rational frame_rate_;
  double duration_ = 0;
  bool hardware_ = false;
  bool draining_ = false;
  double last_time_ = 0;
  bool have_time_ = false;
  AVPacket* packet_ = nullptr;
  AVFrame* frame_ = nullptr;
  AVFrame* props_ = nullptr;
  AVFrame* sw_frame_ = nullptr;
  SwsContext* sws_ = nullptr;
};

}  // namespace advp
