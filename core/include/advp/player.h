// Real-time playback with smoothing: one output frame per display refresh,
// interpolated on the GPU and timed to the refresh it is meant for.
//
// This is the test player for the frame-pacing spike (PLAN.md §8, M0). It
// measures what reaches the screen, so engine choices can be checked against
// real presentation timing rather than benchmarks alone. Video only for now;
// audio and A/V sync arrive with the player core (M1).
#pragma once

#include <cstdint>
#include <string>

#include "advp/engine.h"

namespace advp {

struct PlayOptions {
  std::string input;
  // "auto": benchmark, pick the best engine that fits, adapt at runtime.
  // An engine name or "off": fixed, for comparisons.
  std::string engine = "auto";
  double start = 0;        // seconds into the file
  double max_seconds = 0;  // stop after this long (0: play to the end)
  bool fullscreen = false;
  // Saves one smoothed frame, exactly as displayed, to this PNG path.
  std::string screenshot;
  double screenshot_after = 1.0;  // seconds of playback first
};

struct PlayResult {
  bool ok = false;
  std::string error;

  int width = 0;
  int height = 0;
  std::string decoder;     // "hardware" or "software"
  double source_fps = 0;
  double display_hz = 0;
  std::string start_engine;
  std::string final_engine;
  int engine_changes = 0;

  int64_t refreshes = 0;     // display refreshes while playing
  int64_t presented = 0;     // frames confirmed on screen
  int64_t interpolated = 0;  // of which smoothed in-between frames
  int64_t missed = 0;        // refreshes that showed no new frame
  int64_t underflows = 0;    // refreshes where the decoder was behind
  double work_ms_mean = 0;   // smoothing GPU time per refresh
  double work_ms_p99 = 0;
  double work_ms_max = 0;
  double interval_jitter_ms = 0;  // spread of time between presented frames
  double seconds = 0;
};

// Opens a window and plays until the end, the time limit, or the window is
// closed. Must be called on the main thread.
//
// Keys: space pause, left/right seek 10 s, S smoothing on/off (before/after),
// F full screen, Q or Esc quit.
PlayResult play_file(GpuBackend& backend, const PlayOptions& options);

}  // namespace advp
