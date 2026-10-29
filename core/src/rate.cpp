#include "advp/rate.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

namespace advp {

RateConverter::RateConverter(Rational target_fps, float snap) : target_(target_fps), snap_(snap) {}

double RateConverter::output_time(int64_t index) const {
  return *start_ + static_cast<double>(index) * target_.den / target_.num;
}

std::vector<OutputSlot> RateConverter::push(double source_time) {
  std::vector<OutputSlot> slots;
  if (!start_) {
    start_ = source_time;
    prev_ = source_time;
    return slots;
  }
  // Out-of-order or duplicate timestamps carry no new position; skip them.
  if (source_time <= prev_) return slots;

  const double span = source_time - prev_;
  for (double time = output_time(next_); time < source_time; time = output_time(++next_)) {
    float t = static_cast<float>((time - prev_) / span);
    if (t < snap_) t = 0.f;
    if (t > 1.f - snap_) t = 1.f;
    slots.push_back({next_, time, t});
  }
  prev_ = source_time;
  return slots;
}

std::vector<OutputSlot> RateConverter::flush(double end_time) {
  std::vector<OutputSlot> slots;
  if (!start_) return slots;
  for (double time = output_time(next_); time < end_time; time = output_time(++next_)) {
    slots.push_back({next_, time, 0.f});
  }
  return slots;
}

std::optional<Rational> resolve_target_fps(const char* request, Rational source_fps) {
  if (!request || !*request) return std::nullopt;
  const std::string text(request);
  const bool ntsc = source_fps.den == 1001;

  if (text.back() == 'x') {
    const long factor = std::strtol(text.c_str(), nullptr, 10);
    if (factor < 1 || factor > 8) return std::nullopt;
    return Rational{source_fps.num * factor, source_fps.den};
  }

  char* end = nullptr;
  const double fps = std::strtod(text.c_str(), &end);
  if (end == text.c_str() || *end || fps < 1 || fps > 480) return std::nullopt;

  if (std::floor(fps) == fps) {
    const auto whole = static_cast<int64_t>(fps);
    if (ntsc) return Rational{whole * 1000, 1001};
    return Rational{whole, 1};
  }
  // Explicit NTSC rates such as 59.94 or 119.88.
  const double ntsc_base = std::round(fps * 1.001);
  if (std::fabs(ntsc_base / 1.001 - fps) < 0.01) {
    return Rational{static_cast<int64_t>(ntsc_base) * 1000, 1001};
  }
  return Rational{static_cast<int64_t>(std::llround(fps * 1000)), 1000};
}

}  // namespace advp
