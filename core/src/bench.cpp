#include "advp/bench.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <vector>

#include "json.h"

namespace advp {
namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Textured content panning diagonally, so motion search has real work to do.
void fill_pattern(std::vector<uint8_t>& y, std::vector<uint8_t>& uv, int w, int h, int shift) {
  y.resize(static_cast<size_t>(w) * h);
  uv.resize(static_cast<size_t>(w) * (h / 2));
  for (int row = 0; row < h; ++row) {
    for (int col = 0; col < w; ++col) {
      const int x = col + shift;
      const int yy = row + shift / 2;
      const int v = ((x / 16 + yy / 16) & 1) * 96 + ((x * 7 + yy * 13) & 63) + 48;
      y[static_cast<size_t>(row) * w + col] = static_cast<uint8_t>(v);
    }
  }
  for (int row = 0; row < h / 2; ++row) {
    for (int col = 0; col < w / 2; ++col) {
      uv[static_cast<size_t>(row) * w + col * 2] = static_cast<uint8_t>(128 + ((col + shift / 2) & 31));
      uv[static_cast<size_t>(row) * w + col * 2 + 1] = static_cast<uint8_t>(128 - (row & 31));
    }
  }
}

}  // namespace

std::vector<EngineBenchmark> benchmark_engines(GpuBackend& backend, int width, int height,
                                               int pairs) {
  // A short ring of distinct frames, reused so the benchmark measures the
  // engine and not CPU-side pattern generation.
  constexpr int kRing = 4;
  std::vector<GpuFramePtr> frames;
  std::vector<uint8_t> y, uv;
  for (int i = 0; i < kRing; ++i) {
    auto frame = backend.new_frame(width, height, false);
    fill_pattern(y, uv, width, height, i * 12);
    backend.write_planes(*frame, y.data(), width, uv.data(), width);
    frames.push_back(std::move(frame));
  }
  auto out = backend.new_frame(width, height, false);

  std::vector<EngineBenchmark> results;
  for (const auto& info : backend.engines()) {
    EngineBenchmark r;
    r.engine = info.name;
    r.tier = info.tier;
    r.quality_rank = info.quality_rank;
    r.width = width;
    r.height = height;
    try {
      auto engine = backend.create_engine(info.name);
      // Warm-up: first use compiles pipelines and allocates working memory.
      engine->set_pair(frames[0], frames[1]);
      engine->interpolate(0.5f, *out);

      std::vector<double> pair_ms, interp_ms;
      for (int i = 0; i < pairs; ++i) {
        const auto& a = frames[i % kRing];
        const auto& b = frames[(i + 1) % kRing];
        auto start = Clock::now();
        engine->set_pair(a, b);
        engine->interpolate(0.4f, *out);
        const double first = ms_since(start);
        start = Clock::now();
        engine->interpolate(0.8f, *out);
        const double second = ms_since(start);
        interp_ms.push_back(second);
        pair_ms.push_back(std::max(0.0, first - second));
      }
      // Medians resist one-off hiccups from other apps.
      auto median = [](std::vector<double>& v) {
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return v[v.size() / 2];
      };
      r.pair_ms = median(pair_ms);
      r.interp_ms = median(interp_ms);
      r.ok = true;
    } catch (const std::exception& e) {
      r.error = e.what();
    }
    results.push_back(r);
  }
  return results;
}

double cost_per_second(const EngineBenchmark& bench, double source_fps, double target_fps) {
  const double new_frames = std::max(0.0, target_fps - source_fps);
  return source_fps * bench.pair_ms + new_frames * bench.interp_ms;
}

Selection select_engine(const std::vector<EngineBenchmark>& results, double source_fps,
                        double target_fps, double budget_share) {
  std::vector<const EngineBenchmark*> fitting;
  const double budget_ms = 1000.0 * budget_share;
  for (const auto& r : results) {
    if (r.ok && cost_per_second(r, source_fps, target_fps) <= budget_ms) fitting.push_back(&r);
  }
  std::sort(fitting.begin(), fitting.end(), [](const auto* a, const auto* b) {
    if (a->tier != b->tier) return a->tier > b->tier;
    return a->quality_rank > b->quality_rank;
  });

  Selection s;
  for (const auto* r : fitting) s.ladder.push_back(r->engine);
  s.ladder.push_back("off");
  if (!fitting.empty()) {
    s.engine = fitting.front()->engine;
    s.tier = fitting.front()->tier;
    s.load = cost_per_second(*fitting.front(), source_fps, target_fps) / budget_ms;
  }
  return s;
}

std::string to_json(const std::vector<EngineBenchmark>& results, const Selection& selection,
                    double source_fps, double target_fps) {
  JsonWriter w;
  w.begin_object();
  w.field("source_fps", source_fps);
  w.field("target_fps", target_fps);
  w.key("engines").begin_array();
  for (const auto& r : results) {
    w.begin_object();
    w.field("engine", r.engine);
    w.field("tier", tier_name(r.tier));
    w.field("size", std::to_string(r.width) + "x" + std::to_string(r.height));
    w.field("ok", r.ok);
    if (r.ok) {
      w.field("pair_ms", r.pair_ms);
      w.field("interp_ms", r.interp_ms);
      w.field("gpu_ms_per_second", cost_per_second(r, source_fps, target_fps));
    } else {
      w.field("error", r.error);
    }
    w.end_object();
  }
  w.end_array();
  w.field("selected", selection.engine.empty() ? std::string("off") : selection.engine);
  w.field("selected_load", selection.load);
  w.key("ladder").begin_array();
  for (const auto& name : selection.ladder) w.value(name);
  w.end_array();
  w.end_object();
  return w.str();
}

}  // namespace advp
