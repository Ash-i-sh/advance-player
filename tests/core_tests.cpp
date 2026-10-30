// Unit tests for the platform-independent core. No framework: each check
// prints its failure and the process exits non-zero if any failed.

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "advp/bench.h"
#include "advp/governor.h"
#include "advp/rate.h"
#include "advp/schedule.h"

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
  }
}

bool near(double a, double b, double eps = 1e-6) { return std::fabs(a - b) < eps; }

void test_24_to_60() {
  advp::RateConverter rc({60, 1});
  std::vector<advp::OutputSlot> all;
  for (int i = 0; i <= 24; ++i) {
    auto slots = rc.push(i / 24.0);
    all.insert(all.end(), slots.begin(), slots.end());
  }
  expect(all.size() == 60, "one second of 24 fps yields 60 output frames");
  // 2.5x: positions repeat every two source frames as 0, 0.4, 0.8 | 0.2, 0.6.
  const float expected[] = {0.f, 0.4f, 0.8f, 0.2f, 0.6f};
  for (int i = 0; i < 5; ++i) {
    expect(near(all[i].t, expected[i], 1e-4), "24->60 position " + std::to_string(i));
  }
  for (size_t i = 0; i < all.size(); ++i) {
    expect(all[i].index == static_cast<int64_t>(i), "output indices are consecutive");
  }
}

void test_integer_ratio() {
  advp::RateConverter rc({48, 1});
  std::vector<advp::OutputSlot> all;
  for (int i = 0; i <= 4; ++i) {
    auto slots = rc.push(i / 24.0);
    all.insert(all.end(), slots.begin(), slots.end());
  }
  expect(all.size() == 8, "24->48 doubles the frames");
  for (size_t i = 0; i < all.size(); ++i) {
    expect(near(all[i].t, i % 2 ? 0.5 : 0.0, 1e-4), "24->48 alternates source and midpoint");
  }
}

void test_flush_and_bad_timestamps() {
  advp::RateConverter rc({60, 1});
  rc.push(0.0);
  auto slots = rc.push(0.0);  // duplicate timestamp
  expect(slots.empty(), "duplicate timestamps produce nothing");
  slots = rc.push(1.0 / 24);
  expect(slots.size() == 3, "first 24 fps interval holds 3 output frames at 60 fps");
  slots = rc.flush(0.08);  // just before the 5th output frame at 0.0833 s
  expect(slots.size() == 2, "flush fills the final source frame's duration");
  for (const auto& s : slots) expect(s.t == 0.f, "flush repeats the last frame");
}

void test_variable_frame_rate() {
  advp::RateConverter rc({60, 1});
  size_t total = 0;
  const double times[] = {0.0, 0.040, 0.075, 0.120, 0.166};
  for (double t : times) total += rc.push(t).size();
  expect(total == 10, "VFR source still yields one output frame per 1/60 s");
}

void test_resolve_target() {
  auto ntsc = advp::resolve_target_fps("60", {24000, 1001});
  expect(ntsc && ntsc->num == 60000 && ntsc->den == 1001, "60 on a 23.976 source becomes 59.94");
  auto film = advp::resolve_target_fps("60", {24, 1});
  expect(film && film->num == 60 && film->den == 1, "60 on a 24 source stays 60");
  auto twice = advp::resolve_target_fps("2x", {24000, 1001});
  expect(twice && near(twice->value(), 47.952, 1e-3), "2x doubles the source rate");
  auto explicit_ntsc = advp::resolve_target_fps("119.88", {24, 1});
  expect(explicit_ntsc && explicit_ntsc->num == 120000 && explicit_ntsc->den == 1001,
         "119.88 parses as NTSC 120");
  expect(!advp::resolve_target_fps("fast", {24, 1}), "nonsense is rejected");
  expect(!advp::resolve_target_fps("0", {24, 1}), "zero is rejected");
}

void test_governor_steps_down_and_up() {
  advp::Governor::Config cfg;
  cfg.down_after = 3;
  cfg.up_after = 10;
  advp::Governor g({"motion", "motion-fast", "blend", "off"}, 0, cfg);
  for (int i = 0; i < 3; ++i) g.report(20.0, 16.6);
  expect(g.current() == "motion-fast", "late frames step down one level");
  expect(g.take_changed(), "level change is reported");
  expect(!g.take_changed(), "level change is reported once");
  for (int i = 0; i < 9; ++i) g.report(4.0, 16.6);
  expect(g.current() == "motion-fast", "no step up before the easy streak completes");
  g.report(4.0, 16.6);
  expect(g.current() == "motion", "a long easy streak steps back up");
  g.force_step_down();
  g.force_step_down();
  g.force_step_down();
  g.force_step_down();
  expect(g.current() == "off", "forced steps stop at the bottom of the ladder");
}

void test_governor_ignores_single_spike() {
  advp::Governor::Config cfg;
  cfg.down_after = 3;
  advp::Governor g({"motion", "blend"}, 0, cfg);
  g.report(5.0, 16.6);
  g.report(40.0, 16.6);
  g.report(5.0, 16.6);
  g.report(5.0, 16.6);
  expect(g.current() == "motion", "one slow frame does not change the engine");
}

void test_select_engine() {
  std::vector<advp::EngineBenchmark> results(3);
  results[0] = {"blend", advp::Tier::Blend, 0, 1920, 1080, 0.1, 0.3, true, ""};
  results[1] = {"motion-fast", advp::Tier::Memc, 10, 1920, 1080, 1.0, 1.0, true, ""};
  results[2] = {"motion", advp::Tier::Memc, 20, 1920, 1080, 3.0, 9.0, true, ""};
  // 24 -> 60: motion costs 24*3 + 36*9 = 396 ms/s; budget 500 ms/s.
  auto s = advp::select_engine(results, 24, 60);
  expect(s.engine == "motion", "best engine that fits is chosen");
  expect(s.ladder.size() == 4 && s.ladder.back() == "off", "ladder ends with off");
  // 24 -> 120: motion costs 72 + 96*9 = 936 ms/s, too much; fast fits.
  s = advp::select_engine(results, 24, 120);
  expect(s.engine == "motion-fast", "falls back when the best engine is too slow");
  results[0].ok = results[1].ok = results[2].ok = false;
  s = advp::select_engine(results, 24, 60);
  expect(s.engine.empty() && s.ladder.size() == 1, "no working engine means off");
}

}  // namespace

class FakeFrame final : public advp::GpuFrame {
 public:
  int width() const override { return 16; }
  int height() const override { return 16; }
  bool high_bit_depth() const override { return false; }
  void* native_handle() const override { return nullptr; }
};

void test_schedule_24_on_60hz() {
  advp::FrameSchedule s;
  std::vector<advp::GpuFramePtr> frames;
  for (int i = 0; i < 4; ++i) {
    frames.push_back(std::make_shared<FakeFrame>());
    s.push(frames.back(), i / 24.0);
  }
  // Refreshes at 60 Hz land at 0, 0.4, 0.8, 0.2 (next pair), 0.6...
  const float expected[] = {0.f, 0.4f, 0.8f, 0.2f, 0.6f};
  const int pair[] = {0, 0, 0, 1, 1};
  for (int i = 0; i < 5; ++i) {
    const auto p = s.at(i / 60.0);
    expect(near(p.t, expected[i], 1e-4), "60 Hz position " + std::to_string(i));
    expect(p.a == frames[pair[i]], "60 Hz pair " + std::to_string(i));
    expect(!p.underflow && !p.finished, "60 Hz has frames");
    expect(p.c == frames[pair[i] + 2], "60 Hz look-ahead frame " + std::to_string(i));
  }
  expect(s.size() == 3, "consumed source frames are released");
}

void test_schedule_edges() {
  advp::FrameSchedule s;
  auto a = std::make_shared<FakeFrame>();
  auto b = std::make_shared<FakeFrame>();
  s.push(a, 1.0);
  s.push(b, 1.0);  // duplicate timestamp is dropped
  expect(s.size() == 1, "duplicate timestamp dropped");
  auto p = s.at(0.5);
  expect(p.a == a && p.t == 0.f && !p.underflow, "before the first frame shows it");
  p = s.at(1.0 + 1.0 / 24 + 0.001);
  expect(p.underflow && !p.finished, "next frame overdue is an underflow");
  s.end();
  p = s.at(1.2);
  expect(p.finished && p.a == a, "after the last frame is finished");

  advp::FrameSchedule gap;
  gap.push(a, 0.0);
  gap.push(b, 2.0);  // two-second hole in the stream
  p = gap.at(1.0);
  expect(p.t == 0.f && p.a == a && p.b == a, "gaps hold the earlier frame");
  p = gap.at(2.0);
  expect(p.a == b, "after a gap the later frame shows");
}

// The decoder frees frames the clock has passed while the display is stuck.
void test_schedule_release_before() {
  advp::FrameSchedule s;
  std::vector<std::shared_ptr<FakeFrame>> frames;
  for (int i = 0; i < 10; ++i) {
    frames.push_back(std::make_shared<FakeFrame>());
    s.push(frames.back(), i / 24.0);
  }
  s.release_before(4.5 / 24);
  expect(s.size() == 6 && s.first_time() == 4 / 24.0, "frames before the current one released");
  const auto p = s.at(4.5 / 24);
  expect(p.a == frames[4] && p.b == frames[5], "the current pair is kept");
  s.release_before(100);
  expect(s.size() == 1, "the last frame is always kept");
}

int main() {
  test_24_to_60();
  test_integer_ratio();
  test_flush_and_bad_timestamps();
  test_variable_frame_rate();
  test_resolve_target();
  test_governor_steps_down_and_up();
  test_governor_ignores_single_spike();
  test_select_engine();
  test_schedule_24_on_60hz();
  test_schedule_edges();
  test_schedule_release_before();
  if (failures) {
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("all core tests passed\n");
  return 0;
}
