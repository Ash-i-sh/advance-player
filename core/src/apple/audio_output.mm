#include "audio_output.h"

#include <CoreAudio/CoreAudio.h>
#include <mach/mach_time.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace advp {
namespace {

constexpr double kBufferSeconds = 4.0;
// Beyond this the error is fixed at once (a stall, or a buffer skipped while
// the decoder held the lock); below it, one sample per buffer, which is
// inaudible. Clock drift alone stays under kSlackSeconds.
constexpr double kResyncSeconds = 0.005;
constexpr double kSlackSeconds = 0.001;
// Buffers after a (re)start that are neither counted as underruns nor in the
// sync stats: the first one jumps into place, the rest let the device settle.
constexpr int kSettleBuffers = 8;
// Timestamp gaps or overlaps between decoded chunks smaller than this are
// rounding; larger ones are filled with silence or trimmed.
constexpr double kGapSeconds = 0.01;

void check_os(OSStatus status, const char* what) {
  if (status != noErr) {
    throw std::runtime_error(std::string("audio output: ") + what + " failed (" +
                             std::to_string(status) + ")");
  }
}

template <typename T>
bool device_property(AudioObjectID object, AudioObjectPropertySelector selector,
                     AudioObjectPropertyScope scope, T* value) {
  const AudioObjectPropertyAddress address{selector, scope, kAudioObjectPropertyElementMain};
  UInt32 size = sizeof(T);
  return AudioObjectGetPropertyData(object, &address, 0, nullptr, &size, value) == noErr;
}

}  // namespace

double host_seconds(uint64_t host_time) {
  static const double scale = [] {
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    return static_cast<double>(info.numer) / info.denom / 1e9;
  }();
  return host_time * scale;
}

// ---- MediaClock --------------------------------------------------------------

void MediaClock::run(double origin_host, double origin_media) {
  seq_.fetch_add(1, std::memory_order_acq_rel);
  origin_host_.store(origin_host, std::memory_order_relaxed);
  origin_media_.store(origin_media, std::memory_order_relaxed);
  running_.store(true, std::memory_order_relaxed);
  generation_.fetch_add(1, std::memory_order_relaxed);
  seq_.fetch_add(1, std::memory_order_release);
}

void MediaClock::stop() {
  seq_.fetch_add(1, std::memory_order_acq_rel);
  running_.store(false, std::memory_order_relaxed);
  seq_.fetch_add(1, std::memory_order_release);
}

std::optional<double> MediaClock::at(double host, uint64_t* generation) const {
  while (true) {
    const uint64_t before = seq_.load(std::memory_order_acquire);
    if (before & 1) continue;  // a write is in progress
    const bool running = running_.load(std::memory_order_relaxed);
    const double origin_host = origin_host_.load(std::memory_order_relaxed);
    const double origin_media = origin_media_.load(std::memory_order_relaxed);
    const uint64_t gen = generation_.load(std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (seq_.load(std::memory_order_relaxed) != before) continue;
    if (generation) *generation = gen;
    if (!running) return std::nullopt;
    return origin_media + (host - origin_host);
  }
}

// ---- AudioOutput -------------------------------------------------------------

AudioOutput::AudioOutput(int sample_rate, int channels, const MediaClock& clock)
    : rate_(sample_rate), channels_(channels), clock_(clock) {
  capacity_ = static_cast<size_t>(kBufferSeconds * rate_);
  ring_.assign(capacity_ * channels_, 0.f);

  AudioComponentDescription desc{};
  desc.componentType = kAudioUnitType_Output;
  desc.componentSubType = kAudioUnitSubType_DefaultOutput;  // follows the system's choice
  desc.componentManufacturer = kAudioUnitManufacturer_Apple;
  AudioComponent component = AudioComponentFindNext(nullptr, &desc);
  if (!component) throw std::runtime_error("audio output: no default output device");
  check_os(AudioComponentInstanceNew(component, &unit_), "open device");
  try {
    // Our format; the unit converts to the device's own rate and layout.
    AudioStreamBasicDescription format{};
    format.mSampleRate = rate_;
    format.mFormatID = kAudioFormatLinearPCM;
    format.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
    format.mChannelsPerFrame = static_cast<UInt32>(channels_);
    format.mBitsPerChannel = 32;
    format.mBytesPerFrame = 4 * format.mChannelsPerFrame;
    format.mFramesPerPacket = 1;
    format.mBytesPerPacket = format.mBytesPerFrame;
    check_os(AudioUnitSetProperty(unit_, kAudioUnitProperty_StreamFormat,
                                  kAudioUnitScope_Input, 0, &format, sizeof format),
             "set format");
    AURenderCallbackStruct callback{render_callback, this};
    check_os(AudioUnitSetProperty(unit_, kAudioUnitProperty_SetRenderCallback,
                                  kAudioUnitScope_Input, 0, &callback, sizeof callback),
             "set callback");
    check_os(AudioUnitInitialize(unit_), "initialise");
  } catch (...) {
    AudioComponentInstanceDispose(unit_);
    throw;
  }
  update_latency();
}

AudioOutput::~AudioOutput() {
  stop();
  AudioUnitUninitialize(unit_);
  AudioComponentInstanceDispose(unit_);
}

void AudioOutput::start() {
  if (started_) return;
  check_os(AudioOutputUnitStart(unit_), "start");
  started_ = true;
}

void AudioOutput::stop() {
  close();
  if (!started_) return;
  AudioOutputUnitStop(unit_);  // returns once the render thread is done with us
  started_ = false;
}

void AudioOutput::close() { closed_ = true; }

// Time from the moment the device asks for a buffer until its first sample
// is heard: the device's latency, its I/O buffer and safety offset (counted
// as mpv does), plus any delay of the unit's own rate conversion.
void AudioOutput::update_latency() {
  AudioDeviceID device = 0;
  UInt32 size = sizeof device;
  if (AudioUnitGetProperty(unit_, kAudioOutputUnitProperty_CurrentDevice,
                           kAudioUnitScope_Global, 0, &device, &size) != noErr) {
    return;
  }
  Float64 device_rate = 0;
  if (!device_property(device, kAudioDevicePropertyNominalSampleRate,
                       kAudioObjectPropertyScopeGlobal, &device_rate) ||
      device_rate <= 0) {
    return;
  }
  UInt32 frames = 0;
  const AudioObjectPropertySelector parts[] = {kAudioDevicePropertyLatency,
                                               kAudioDevicePropertyBufferFrameSize,
                                               kAudioDevicePropertySafetyOffset};
  for (AudioObjectPropertySelector s : parts) {
    UInt32 value = 0;
    if (device_property(device, s, kAudioDevicePropertyScopeOutput, &value)) frames += value;
  }
  Float64 unit_latency = 0;
  size = sizeof unit_latency;
  AudioUnitGetProperty(unit_, kAudioUnitProperty_Latency, kAudioUnitScope_Global, 0,
                       &unit_latency, &size);
  latency_ = frames / device_rate + unit_latency;
}

void AudioOutput::set_timeline(const Timeline& timeline) {
  timeline_seq_.fetch_add(1, std::memory_order_acq_rel);
  timeline_start_.store(timeline.start, std::memory_order_relaxed);
  timeline_base_.store(timeline.base, std::memory_order_relaxed);
  timeline_origin_.store(timeline.origin, std::memory_order_relaxed);
  timeline_seq_.fetch_add(1, std::memory_order_release);
  producer_timeline_ = timeline;
}

AudioOutput::Timeline AudioOutput::timeline() const {
  return Timeline{timeline_start_.load(std::memory_order_relaxed),
                  timeline_base_.load(std::memory_order_relaxed),
                  timeline_origin_.load(std::memory_order_relaxed)};
}

void AudioOutput::push(const float* samples, int frames, double time,
                       const std::function<bool()>& give_up) {
  uint64_t write = write_pos_.load(std::memory_order_relaxed);
  const auto used = [&] {
    return write - std::max(read_pos_.load(std::memory_order_acquire), producer_timeline_.start);
  };
  if (used() == 0) {
    // Nothing buffered: this chunk starts a new stretch of the timeline.
    set_timeline(Timeline{write, write, time});
  } else {
    const Timeline& tl = producer_timeline_;
    const double end = tl.origin + static_cast<double>(write - tl.base) / rate_;
    const double gap = time - end;
    if (gap > kGapSeconds) {
      // Missing audio (damaged or sparse stream): keep the timeline with silence.
      const uint64_t silence =
          std::min<uint64_t>(capacity_ - used(), static_cast<uint64_t>(std::llround(gap * rate_)));
      for (uint64_t i = 0; i < silence; ++i) {
        std::fill_n(&ring_[((write + i) % capacity_) * channels_], channels_, 0.f);
      }
      write += silence;
      write_pos_.store(write, std::memory_order_release);
    } else if (gap < -kGapSeconds) {
      const int skip = static_cast<int>(std::llround(-gap * rate_));
      if (skip >= frames) return;
      samples += static_cast<size_t>(skip) * channels_;
      frames -= skip;
    }
  }
  while (frames > 0) {
    const uint64_t room = capacity_ - used();
    if (room == 0) {
      // Full: playback is far behind the decoder. Wait for room.
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      if (closed_ || give_up()) return;
      continue;
    }
    const uint64_t at = write % capacity_;
    const uint64_t n = std::min<uint64_t>({static_cast<uint64_t>(frames), room, capacity_ - at});
    std::memcpy(&ring_[at * channels_], samples, n * channels_ * sizeof(float));
    write += n;
    write_pos_.store(write, std::memory_order_release);
    samples += n * channels_;
    frames -= static_cast<int>(n);
  }
}

void AudioOutput::reset() {
  // Everything written so far is stale; the next push starts a new timeline.
  const uint64_t write = write_pos_.load(std::memory_order_relaxed);
  set_timeline(Timeline{write, write, 0});
  ended_ = false;
}

double AudioOutput::buffered() const {
  const uint64_t write = write_pos_.load(std::memory_order_relaxed);
  const uint64_t read =
      std::max(read_pos_.load(std::memory_order_acquire), producer_timeline_.start);
  return write > read ? static_cast<double>(write - read) / rate_ : 0.0;
}

AudioOutputStats AudioOutput::stats() const {
  AudioOutputStats s;
  s.underruns = underruns_;
  s.resyncs = resyncs_;
  s.sync_ms_mean = sync_count_ ? sync_sum_ / sync_count_ * 1000.0 : 0.0;
  s.sync_ms_max = sync_max_ * 1000.0;
  return s;
}

OSStatus AudioOutput::render_callback(void* context, AudioUnitRenderActionFlags*,
                                      const AudioTimeStamp* time, UInt32, UInt32 frames,
                                      AudioBufferList* data) {
  auto* self = static_cast<AudioOutput*>(context);
  auto* out = static_cast<float*>(data->mBuffers[0].mData);
  std::fill_n(out, static_cast<size_t>(frames) * self->channels_, 0.f);
  self->render(time, frames, out);
  return noErr;
}

void AudioOutput::copy_out(uint64_t from, float* out, size_t frames) const {
  for (size_t done = 0; done < frames;) {
    const uint64_t at = (from + done) % capacity_;
    const size_t n = std::min<size_t>(frames - done, capacity_ - at);
    std::memcpy(out + done * channels_, &ring_[at * channels_], n * channels_ * sizeof(float));
    done += n;
  }
}

// Runs on Core Audio's real-time thread: no locks, no waiting, no
// allocation. out is already silent; anything not filled stays so.
void AudioOutput::render(const AudioTimeStamp* time, UInt32 frames, float* out) {
  if (!(time->mFlags & kAudioTimeStampHostTimeValid)) return;
  uint64_t generation = 0;
  const std::optional<double> media =
      clock_.at(host_seconds(time->mHostTime) + latency_.load(), &generation);
  if (!media) return;  // paused or not started
  if (generation != generation_) {
    generation_ = generation;
    settle_buffers_ = kSettleBuffers;
  }

  // The timeline changes only on a reset or a fresh start, when there is
  // nothing worth playing; if it changes under us, this buffer stays silent.
  const uint64_t seq = timeline_seq_.load(std::memory_order_acquire);
  if (seq & 1) return;
  const Timeline tl = timeline();
  const uint64_t write = write_pos_.load(std::memory_order_acquire);
  uint64_t read = std::max(read_pos_.load(std::memory_order_relaxed), tl.start);
  uint64_t available = write > read ? write - read : 0;
  const auto finish = [&](uint64_t new_read) {
    if (timeline_seq_.load(std::memory_order_acquire) != seq) {
      std::fill_n(out, static_cast<size_t>(frames) * channels_, 0.f);
      return;
    }
    read_pos_.store(new_read, std::memory_order_release);
  };
  if (available == 0) {
    if (settle_buffers_ == 0 && !ended_) ++underruns_;
    finish(read);
    return;
  }
  const auto drop = [&](uint64_t n) {
    n = std::min(n, available);
    read += n;
    available -= n;
  };

  size_t filled = 0;
  const double head_time = tl.origin + static_cast<double>(static_cast<int64_t>(read - tl.base)) / rate_;
  const double error = head_time - *media;  // > 0: the buffered audio is for later
  // Line up exactly on the first buffer after a (re)start; later, only a
  // large error is fixed at once.
  const bool restart = settle_buffers_ == kSettleBuffers;
  if (std::fabs(error) > (restart ? kSlackSeconds : kResyncSeconds)) {
    if (settle_buffers_ == 0) ++resyncs_;
    settle_buffers_ = kSettleBuffers;
    if (error < 0) {
      drop(static_cast<uint64_t>(std::llround(-error * rate_)));
    } else {
      filled = std::min<size_t>(frames, static_cast<size_t>(std::llround(error * rate_)));
    }
  } else {
    if (settle_buffers_ > 0) {
      --settle_buffers_;
    } else {
      sync_sum_ += std::fabs(error);
      sync_max_ = std::max(sync_max_, std::fabs(error));
      ++sync_count_;
    }
    if (error < -kSlackSeconds && available > 1) {
      drop(1);
    } else if (error > kSlackSeconds && available > 0) {
      copy_out(read, out, 1);  // play the next frame twice
      filled = 1;
    }
  }
  const size_t want = frames - filled;
  const size_t got = static_cast<size_t>(std::min<uint64_t>(want, available));
  copy_out(read, out + filled * channels_, got);
  read += got;
  if (got < want && settle_buffers_ == 0 && !ended_) ++underruns_;

  const float volume = volume_.load();
  if (volume != 1.f) {
    for (size_t i = 0, n = static_cast<size_t>(frames) * channels_; i < n; ++i) out[i] *= volume;
  }
  finish(read);
}

}  // namespace advp
