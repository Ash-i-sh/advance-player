// Runtime quality governor: steps down the engine ladder when frames run late
// and back up when there is headroom. It knows nothing about devices; it only
// sees measured work time against each frame's deadline.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace advp {

class Governor {
 public:
  struct Config {
    double step_down_load = 0.90;  // smoothed load above this is "falling behind"
    double step_up_load = 0.45;    // below this is "plenty of headroom"
    int down_after = 6;            // consecutive late frames before stepping down
    int up_after = 300;            // consecutive easy frames before stepping up
    double smoothing = 0.15;       // EWMA weight of the newest sample
  };

  // ladder: engine names, best first. start: index to begin at.
  Governor(std::vector<std::string> ladder, size_t start);
  Governor(std::vector<std::string> ladder, size_t start, Config config);

  // Reports one processed frame: work_ms spent against budget_ms available.
  void report(double work_ms, double budget_ms);
  // Thermal pressure or battery saver: drop a step immediately.
  void force_step_down();

  size_t level() const { return level_; }
  const std::string& current() const { return ladder_[level_]; }
  double load() const { return load_; }
  // True once after every level change.
  bool take_changed();

 private:
  void move_to(size_t level);

  std::vector<std::string> ladder_;
  Config config_;
  size_t level_;
  double load_ = 0;
  bool have_load_ = false;
  int late_ = 0;
  int easy_ = 0;
  bool changed_ = false;
};

}  // namespace advp
