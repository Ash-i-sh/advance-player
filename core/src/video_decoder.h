// Opens a media file and decodes its main video stream into GPU frames,
// using the hardware decoder when the stream allows it. Shared by the
// converter and the player.
#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "advp/engine.h"
#include "advp/rate.h"

struct AVFormatContext;
struct AVStream;
struct AVFrame;
struct AVPacket;
struct AVCodecContext;
struct SwsContext;

namespace advp {

// FFmpeg options for opening an input, such as HTTP headers for web streams.
using OpenOptions = std::map<std::string, std::string>;

// Opens path (a file or URL) with options; throws on failure.
AVFormatContext* open_input(const std::string& path, const OpenOptions& options);

struct DecodedFrame {
  GpuFramePtr frame;
  double time = 0;  // presentation time in seconds
};

class VideoDecoder {
 public:
  // read_ahead: read packets on a thread of its own, up to tens of seconds
  // ahead, so network stalls do not reach the picture (for web streams).
  VideoDecoder(GpuBackend& backend, const std::string& path, const OpenOptions& options = {},
               bool read_ahead = false);
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

  // With read-ahead: seconds of video read but not yet decoded, and whether
  // reading has reached the end. Safe from any thread.
  double read_ahead_seconds();
  bool read_ahead_done();

 private:
  void release();
  GpuFramePtr to_gpu(AVFrame* frame);
  // The next packet of any stream into packet; false at the end.
  bool read_packet(AVPacket* packet);
  void read_ahead_loop();
  double packet_seconds(const AVPacket* packet) const;  // video stream

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

  // Read-ahead: the thread owns format_ reads and seeks; next() takes from
  // the queue.
  std::thread reader_;
  std::mutex read_mu_;
  std::condition_variable read_cv_;
  std::deque<AVPacket*> queue_;
  size_t queued_bytes_ = 0;
  double queued_seconds_ = 0;  // video packet durations
  bool read_end_ = false;
  bool read_stop_ = false;
  std::optional<int64_t> read_seek_;  // video stream timestamp
};

}  // namespace advp
