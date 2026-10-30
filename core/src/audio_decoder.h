// Decodes one audio stream to interleaved 32-bit float stereo at the stream's
// own sample rate (surround is downmixed). Packets come from the demuxer that
// VideoDecoder owns, so audio and video are read in one pass.
#pragma once

#include <functional>
#include <string>

struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct AVStream;
struct SwrContext;

namespace advp {

struct AudioChunk {
  const float* samples = nullptr;  // interleaved, channels() per frame
  int frames = 0;
  double time = 0;  // presentation time of the first frame, seconds
};

class AudioDecoder {
 public:
  explicit AudioDecoder(AVStream* stream);
  ~AudioDecoder();
  AudioDecoder(const AudioDecoder&) = delete;
  AudioDecoder& operator=(const AudioDecoder&) = delete;

  int stream_index() const { return index_; }
  int sample_rate() const { return rate_; }
  static constexpr int channels() { return 2; }
  // For example "opus, 48000 Hz, 6 ch".
  std::string description() const;

  // Decodes one packet of this stream; sink gets each decoded chunk, valid
  // only during the call.
  void decode(const AVPacket* packet, const std::function<void(const AudioChunk&)>& sink);

  // Forgets buffered state, after a seek.
  void flush();

 private:
  void convert(AVFrame* frame, const std::function<void(const AudioChunk&)>& sink);

  AVCodecContext* codec_ = nullptr;
  AVFrame* frame_ = nullptr;
  SwrContext* swr_ = nullptr;
  int swr_format_ = -1;
  int swr_channels_ = 0;
  int index_ = -1;
  int rate_ = 0;
  double time_base_ = 0;
  double next_time_ = 0;
  bool have_time_ = false;
  float* out_ = nullptr;
  int out_capacity_ = 0;  // frames
};

}  // namespace advp
