// Chooses which source frames to show, and where between them, for a
// playback time. The decoder pushes frames in; the display asks once per
// refresh. Not thread-safe: the player guards it with its own lock.
#pragma once

#include <cstddef>
#include <deque>

#include "advp/engine.h"

namespace advp {

struct ScheduledPair {
  GpuFramePtr a;           // source frame at or before the playback time
  GpuFramePtr b;           // the one after it; equals a when there is none
  GpuFramePtr c;           // the one after b if decoded, so its pair can be analysed early
  float t = 0.f;           // 0 = a, 1 = b; 0 also when interpolating makes no sense
  double a_time = 0;
  bool underflow = false;  // the next source frame is due but not decoded yet
  bool finished = false;   // past the last frame of the stream
};

class FrameSchedule {
 public:
  // snap: positions this close to a source frame show it unchanged.
  // max_gap: source frames further apart than this (seconds) are a gap in the
  // stream, not motion, so the earlier frame is held instead of morphed.
  explicit FrameSchedule(float snap = 0.02f, double max_gap = 0.25);

  // Frames must arrive in presentation order; out-of-order or repeated
  // timestamps are dropped.
  void push(GpuFramePtr frame, double time);
  // No more frames will come.
  void end() { ended_ = true; }
  // Drops everything, e.g. after a seek.
  void clear();

  size_t size() const { return frames_.size(); }
  bool ended() const { return ended_; }
  bool empty() const { return frames_.empty(); }
  double first_time() const { return frames_.empty() ? 0.0 : frames_.front().time; }

  // The pair for playback time `time`. Frames no longer needed are released.
  ScheduledPair at(double time);
  // Releases frames that playback time `time` has passed, as at() would,
  // without choosing a pair. Lets the decoder make room while the display is
  // held up, instead of stopping (and starving audio read from the same file).
  void release_before(double time);

 private:
  struct Entry {
    GpuFramePtr frame;
    double time;
  };
  std::deque<Entry> frames_;
  float snap_;
  double max_gap_;
  double frame_duration_ = 1.0 / 24;  // last observed spacing
  bool ended_ = false;
};

}  // namespace advp
