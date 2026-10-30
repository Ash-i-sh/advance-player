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

void AudioOutput::push(const float* samples, int frames, double time,
                       const std::function<bool()>& give_up) {
  std::unique_lock<std::mutex> lock(mu_);
  if (count_ == 0) {
    head_time_ = time;
  } else {
    const double gap = time - (head_time_ + static_cast<double>(count_) / rate_);
    if (gap > kGapSeconds) {
      // Missing audio (damaged or sparse stream): keep the timeline with silence.
      size_t silence = std::min(capacity_ - count_, static_cast<size_t>(std::llround(gap * rate_)));
      for (; silence > 0; --silence) {
        std::fill_n(&ring_[((read_ + count_) % capacity_) * channels_], channels_, 0.f);
        ++count_;
      }
    } else if (gap < -kGapSeconds) {
      const int skip = static_cast<int>(std::llround(-gap * rate_));
      if (skip >= frames) return;
      samples += static_cast<size_t>(skip) * channels_;
      frames -= skip;
    }
  }
  while (frames > 0) {
    if (count_ == capacity_) {
      // Full: playback is far behind the decoder (paused). Wait for room.
      lock.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      if (closed_ || give_up()) return;
      lock.lock();
      if (count_ == 0) return;  // reset meanwhile; this audio is stale
      continue;
    }
    const size_t at = (read_ + count_) % capacity_;
    const size_t n = std::min({static_cast<size_t>(frames), capacity_ - count_, capacity_ - at});
    std::memcpy(&ring_[at * channels_], samples, n * channels_ * sizeof(float));
    count_ += n;
    samples += n * channels_;
    frames -= static_cast<int>(n);
  }
}

void AudioOutput::reset() {
  std::lock_guard<std::mutex> lock(mu_);
  read_ = 0;
  count_ = 0;
}

double AudioOutput::buffered() const {
  std::lock_guard<std::mutex> lock(mu_);
  return static_cast<double>(count_) / rate_;
}

AudioOutputStats AudioOutput::stats() const {
  std::lock_guard<std::mutex> lock(mu_);
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

void AudioOutput::drop_front(size_t frames) {
  frames = std::min(frames, count_);
  read_ = (read_ + frames) % capacity_;
  count_ -= frames;
  head_time_ += static_cast<double>(frames) / rate_;
}

size_t AudioOutput::copy_front(float* out, size_t frames) {
  frames = std::min(frames, count_);
  for (size_t done = 0; done < frames;) {
    const size_t n = std::min(frames - done, capacity_ - read_);
    std::memcpy(out + done * channels_, &ring_[read_ * channels_], n * channels_ * sizeof(float));
    read_ = (read_ + n) % capacity_;
    done += n;
  }
  count_ -= frames;
  head_time_ += static_cast<double>(frames) / rate_;
  return frames;
}

// Runs on Core Audio's real-time thread: no waiting, no allocation. out is
// already silent; anything not filled stays so.
void AudioOutput::render(const AudioTimeStamp* time, UInt32 frames, float* out) {
  if (!(time->mFlags & kAudioTimeStampHostTimeValid)) return;
  uint64_t generation = 0;
  const std::optional<double> media =
      clock_.at(host_seconds(time->mHostTime) + latency_.load(), &generation);
  if (!media) return;  // paused or not started
  std::unique_lock<std::mutex> lock(mu_, std::try_to_lock);
  if (!lock.owns_lock()) return;  // the decoder holds it for a moment

  if (generation != generation_) {
    generation_ = generation;
    settle_buffers_ = kSettleBuffers;
  }
  if (count_ == 0) {
    if (settle_buffers_ == 0) ++underruns_;
    return;
  }

  size_t filled = 0;
  const double error = head_time_ - *media;  // > 0: the buffered audio is for later
  // Line up exactly on the first buffer after a (re)start; later, only a
  // large error is fixed at once.
  const bool restart = settle_buffers_ == kSettleBuffers;
  if (std::fabs(error) > (restart ? kSlackSeconds : kResyncSeconds)) {
    if (settle_buffers_ == 0) ++resyncs_;
    settle_buffers_ = kSettleBuffers;
    if (error < 0) {
      drop_front(static_cast<size_t>(std::llround(-error * rate_)));
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
    if (error < -kSlackSeconds && count_ > 1) {
      drop_front(1);
    } else if (error > kSlackSeconds) {
      std::memcpy(out, &ring_[read_ * channels_], channels_ * sizeof(float));  // repeat one
      filled = 1;
    }
  }
  const size_t want = frames - filled;
  if (copy_front(out + filled * channels_, want) < want && settle_buffers_ == 0) ++underruns_;

  const float volume = volume_.load();
  if (volume != 1.f) {
    for (size_t i = 0, n = static_cast<size_t>(frames) * channels_; i < n; ++i) out[i] *= volume;
  }
}

}  // namespace advp
