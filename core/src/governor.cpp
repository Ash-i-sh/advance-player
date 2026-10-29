#include "advp/governor.h"

#include <utility>

namespace advp {

Governor::Governor(std::vector<std::string> ladder, size_t start)
    : Governor(std::move(ladder), start, Config{}) {}

Governor::Governor(std::vector<std::string> ladder, size_t start, Config config)
    : ladder_(std::move(ladder)), config_(config) {
  if (ladder_.empty()) ladder_.push_back("off");
  level_ = start < ladder_.size() ? start : ladder_.size() - 1;
}

void Governor::report(double work_ms, double budget_ms) {
  if (budget_ms <= 0) return;
  const double sample = work_ms / budget_ms;
  load_ = have_load_ ? load_ + config_.smoothing * (sample - load_) : sample;
  have_load_ = true;

  // A single frame far over budget counts as late even before the average
  // catches up, so a sudden heavy scene reacts within a few frames.
  if (load_ > config_.step_down_load || sample > 1.5) {
    easy_ = 0;
    if (++late_ >= config_.down_after && level_ + 1 < ladder_.size()) move_to(level_ + 1);
  } else if (load_ < config_.step_up_load) {
    late_ = 0;
    if (++easy_ >= config_.up_after && level_ > 0) move_to(level_ - 1);
  } else {
    late_ = 0;
    easy_ = 0;
  }
}

void Governor::force_step_down() {
  if (level_ + 1 < ladder_.size()) move_to(level_ + 1);
}

bool Governor::take_changed() {
  return std::exchange(changed_, false);
}

void Governor::move_to(size_t level) {
  level_ = level;
  late_ = 0;
  easy_ = 0;
  have_load_ = false;  // the new engine has a different cost; start fresh
  changed_ = true;
}

}  // namespace advp
