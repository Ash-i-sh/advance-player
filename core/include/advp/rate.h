// Maps output frame times onto source frame pairs.
//
// Works from real timestamps, so variable-frame-rate sources and non-integer
// ratios (24 -> 60 is 2.5x) need no special cases.
#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace advp {

struct Rational {
  int64_t num = 0;
  int64_t den = 1;
  double value() const { return den ? static_cast<double>(num) / den : 0.0; }
};

// One output frame. t is its position between the previous source frame (0)
// and the newest one (1); 0 and 1 mean "repeat that source frame as-is".
struct OutputSlot {
  int64_t index = 0;  // output frame number, 0-based
  double time = 0;    // seconds
  float t = 0;
};

class RateConverter {
 public:
  // snap: positions closer than this to a source frame reuse it unchanged.
  explicit RateConverter(Rational target_fps, float snap = 0.02f);

  // Feed source frames in presentation order. Returns the output frames that
  // fall between the previous source frame and this one.
  std::vector<OutputSlot> push(double source_time);
  // Emits the remaining output frames up to end_time, repeating the last frame.
  std::vector<OutputSlot> flush(double end_time);

  double start_time() const { return start_.value_or(0.0); }

 private:
  double output_time(int64_t index) const;

  Rational target_;
  float snap_;
  std::optional<double> start_;
  double prev_ = 0;
  int64_t next_ = 0;
};

// Picks the output rate for a requested target ("60", "120", "59.94", "2x").
// An integer target keeps NTSC timing when the source uses it, so 23.976 fps
// becomes 59.94, not 60, and the 2.5x ratio stays exact.
std::optional<Rational> resolve_target_fps(const char* request, Rational source_fps);

}  // namespace advp
