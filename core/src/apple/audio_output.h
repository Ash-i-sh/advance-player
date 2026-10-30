// macOS audio output that follows the player's video clock.
//
// Video timing is the master: frames are tied to display refreshes, which
// keeps their pacing exact. Audio follows it: for every buffer the device
// asks for, the output works out which media time that buffer will be heard
// at (from its host time plus the device's latency) and plays the samples
// for that time. Small drift between the audio and display clocks is taken
// out one sample at a time; a large error (start, seek) is fixed at once by
// dropping samples or waiting with silence.
#pragma once

#include <AudioToolbox/AudioToolbox.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace advp {

// Media time as a function of host time, readable from the audio thread
// without locks. Written by the display thread when the clock (re)starts,
// pauses or stops.
class MediaClock {
 public:
  void run(double origin_host, double origin_media);
  void stop();
  // Media time at host_seconds, or nothing while stopped or paused.
  // generation changes on every run(), so readers can tell restarts apart.
  std::optional<double> at(double host_seconds, uint64_t* generation = nullptr) const;

 private:
  std::atomic<uint64_t> seq_{0};  // odd while being written
  std::atomic<double> origin_host_{0};
  std::atomic<double> origin_media_{0};
  std::atomic<bool> running_{false};
  std::atomic<uint64_t> generation_{0};
};

double host_seconds(uint64_t host_time);

struct AudioOutputStats {
  int64_t underruns = 0;  // buffers that ran out of decoded audio while playing
  int64_t resyncs = 0;    // large corrections after playback had settled
  double sync_ms_mean = 0;  // |audio - video clock| while settled
  double sync_ms_max = 0;
};

class AudioOutput {
 public:
  AudioOutput(int sample_rate, int channels, const MediaClock& clock);
  ~AudioOutput();
  AudioOutput(const AudioOutput&) = delete;
  AudioOutput& operator=(const AudioOutput&) = delete;

  void start();
  void stop();

  // Producer (decode thread). Blocks while the buffer is full, until there
  // is room or give_up() says the audio is no longer wanted.
  void push(const float* samples, int frames, double time, const std::function<bool()>& give_up);
  // Drops all buffered audio (seek).
  void reset();
  // Audio buffered ahead, in seconds.
  double buffered() const;
  // Ends a blocked push() for good (shutdown).
  void close();

  void set_volume(float volume) { volume_ = volume; }
  float volume() const { return volume_; }
  // Re-reads the output device's latency (main thread; the default device
  // can change while playing).
  void update_latency();
  double latency() const { return latency_; }

  AudioOutputStats stats() const;

 private:
  static OSStatus render_callback(void* context, AudioUnitRenderActionFlags* flags,
                                  const AudioTimeStamp* time, UInt32 bus, UInt32 frames,
                                  AudioBufferList* data);
  void render(const AudioTimeStamp* time, UInt32 frames, float* out);
  void drop_front(size_t frames);
  size_t copy_front(float* out, size_t frames);

  const int rate_;
  const int channels_;
  const MediaClock& clock_;
  AudioUnit unit_ = nullptr;
  bool started_ = false;

  std::atomic<float> volume_{1.f};
  std::atomic<double> latency_{0};
  std::atomic<bool> closed_{false};

  // Ring buffer of interleaved samples whose first frame plays at head_time_.
  // The audio thread only try-locks, so it never waits on the decoder.
  mutable std::mutex mu_;
  std::vector<float> ring_;
  size_t capacity_ = 0;  // frames
  size_t read_ = 0;      // frame index of the head
  size_t count_ = 0;     // frames buffered
  double head_time_ = 0;

  // Audio thread state and stats (stats read under mu_).
  uint64_t generation_ = UINT64_MAX;
  int settle_buffers_ = 0;
  int64_t underruns_ = 0;
  int64_t resyncs_ = 0;
  double sync_sum_ = 0;
  double sync_max_ = 0;
  int64_t sync_count_ = 0;
};

}  // namespace advp
