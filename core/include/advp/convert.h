// Offline conversion: decode -> smooth -> hardware encode, keeping audio and
// subtitle tracks.
#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "advp/engine.h"

namespace advp {

struct ConvertProgress {
  int64_t source_frames = 0;
  int64_t output_frames = 0;
  double media_seconds = 0;   // position in the source
  double duration = 0;        // total, 0 if unknown
  double elapsed = 0;         // wall clock
};

struct ConvertOptions {
  std::string input;
  std::string output;
  std::string target_fps = "60";  // "60", "120", "59.94", "2x"
  std::string engine = "auto";    // "auto" or an engine name
  std::string codec = "hevc";     // "hevc" or "h264"
  int quality = 70;               // 1..100 constant quality
  int64_t max_source_frames = 0;  // 0 = whole file
  std::function<void(const ConvertProgress&)> on_progress;
};

struct ConvertResult {
  bool ok = false;
  std::string error;
  std::string engine;
  std::string decoder;            // "hardware" or "software"
  double source_fps = 0;
  double target_fps = 0;
  int64_t source_frames = 0;
  int64_t output_frames = 0;
  int64_t interpolated = 0;
  int64_t scene_cuts = 0;
  double seconds = 0;
};

ConvertResult convert_file(GpuBackend& backend, const ConvertOptions& options);

}  // namespace advp
