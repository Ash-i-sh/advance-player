#include "advp/schedule.h"

#include <utility>

namespace advp {

FrameSchedule::FrameSchedule(float snap, double max_gap) : snap_(snap), max_gap_(max_gap) {}

void FrameSchedule::push(GpuFramePtr frame, double time) {
  if (!frames_.empty()) {
    const double spacing = time - frames_.back().time;
    if (spacing <= 0) return;
    if (spacing <= max_gap_) frame_duration_ = spacing;
  }
  frames_.push_back({std::move(frame), time});
}

void FrameSchedule::clear() {
  frames_.clear();
  ended_ = false;
}

ScheduledPair FrameSchedule::at(double time) {
  ScheduledPair p;
  if (frames_.empty()) {
    p.finished = ended_;
    p.underflow = !ended_;
    return p;
  }
  while (frames_.size() >= 2 && frames_[1].time <= time) frames_.pop_front();

  const Entry& a = frames_.front();
  p.a = a.frame;
  p.b = a.frame;
  p.a_time = a.time;
  if (frames_.size() == 1) {
    // Last frame we have: hold it. Past its duration it is either the end of
    // the stream or the decoder falling behind.
    if (time >= a.time + frame_duration_) {
      p.finished = ended_;
      p.underflow = !ended_;
    }
    return p;
  }
  const Entry& b = frames_[1];
  const double span = b.time - a.time;
  if (span > max_gap_) return p;
  p.b = b.frame;
  if (frames_.size() >= 3 && frames_[2].time - b.time <= max_gap_) p.c = frames_[2].frame;
  if (time <= a.time) return p;  // at or before the first frame
  const float t = static_cast<float>((time - a.time) / span);
  if (t < snap_) {
    p.t = 0.f;
  } else if (t > 1.f - snap_) {
    p.t = 1.f;
  } else {
    p.t = t;
  }
  return p;
}

}  // namespace advp
