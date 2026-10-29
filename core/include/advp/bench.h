// Measures each engine on this device and picks the best one that fits the
// real-time budget.
#pragma once

#include <string>
#include <vector>

#include "advp/engine.h"

namespace advp {

struct EngineBenchmark {
  std::string engine;
  Tier tier = Tier::Off;
  int quality_rank = 0;
  int width = 0;
  int height = 0;
  double pair_ms = 0;    // analysing one new source pair
  double interp_ms = 0;  // producing one in-between frame
  bool ok = false;
  std::string error;
};

// Runs every engine on synthetic moving content at the given size.
std::vector<EngineBenchmark> benchmark_engines(GpuBackend& backend, int width, int height,
                                               int pairs = 24);

// Milliseconds of GPU work per second of playback for this engine.
double cost_per_second(const EngineBenchmark& bench, double source_fps, double target_fps);

struct Selection {
  std::string engine;  // empty => no engine fits; play without smoothing
  Tier tier = Tier::Off;
  double load = 0;     // share of the budget the engine would use
  std::vector<std::string> ladder;  // engines that fit, best first, for the governor
};

// budget_share: part of each second the engine may use; the rest is left for
// decoding, colour processing, presentation and the UI.
Selection select_engine(const std::vector<EngineBenchmark>& results, double source_fps,
                        double target_fps, double budget_share = 0.5);

std::string to_json(const std::vector<EngineBenchmark>& results, const Selection& selection,
                    double source_fps, double target_fps);

}  // namespace advp
