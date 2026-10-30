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

  // Producer side: push(), reset() and buffered() are for one thread (the
  // decoder). push() blocks while the buffer is full, until there is room or
  // give_up() says the audio is no longer wanted.
  void push(const float* samples, int frames, double time, const std::function<bool()>& give_up);
  // Drops all buffered audio (seek).
  void reset();
  // No more audio is coming (end of file): running out is not an underrun.
  void end() { ended_ = true; }
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

  // Read after stop().
  AudioOutputStats stats() const;

 private:
  // Where buffered audio sits in media time: the frame at ring position p
  // plays at origin + (p - base) / rate. Frames before start are stale
  // (written before a reset). Written by the producer, read by the audio
  // thread without locks (sequence counter, odd while being written).
  struct Timeline {
    uint64_t start;
    uint64_t base;
    double origin;
  };
  void set_timeline(const Timeline& timeline);
  Timeline timeline() const;

  static OSStatus render_callback(void* context, AudioUnitRenderActionFlags* flags,
                                  const AudioTimeStamp* time, UInt32 bus, UInt32 frames,
                                  AudioBufferList* data);
  void render(const AudioTimeStamp* time, UInt32 frames, float* out);
  void copy_out(uint64_t from, float* out, size_t frames) const;

  const int rate_;
  const int channels_;
  const MediaClock& clock_;
  AudioUnit unit_ = nullptr;
  bool started_ = false;

  std::atomic<float> volume_{1.f};
  std::atomic<double> latency_{0};
  std::atomic<bool> closed_{false};
  std::atomic<bool> ended_{false};

  // Single-producer, single-consumer ring of interleaved samples. Positions
  // count frames since the start and only grow; the audio thread never
  // waits for the decoder, and the decoder only waits when the ring is full.
  std::vector<float> ring_;
  uint64_t capacity_ = 0;  // frames
  std::atomic<uint64_t> write_pos_{0};  // producer
  std::atomic<uint64_t> read_pos_{0};   // audio thread
  std::atomic<uint64_t> timeline_seq_{0};
  std::atomic<uint64_t> timeline_start_{0};
  std::atomic<uint64_t> timeline_base_{0};
  std::atomic<double> timeline_origin_{0};
  Timeline producer_timeline_{};  // the producer's own copy

  // Audio thread only.
  uint64_t generation_ = UINT64_MAX;
  int settle_buffers_ = 0;
  int64_t underruns_ = 0;
  int64_t resyncs_ = 0;
  double sync_sum_ = 0;
  double sync_max_ = 0;
  int64_t sync_count_ = 0;
};

}  // namespace advp
