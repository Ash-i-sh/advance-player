// Describes what the current device can do. Gathered at runtime, never
// hard-coded per model; engine choice is then driven by benchmarks.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace advp {

struct DisplayInfo {
  std::string name;
  int width = 0;
  int height = 0;
  double refresh_hz = 0;
  bool hdr = false;
  bool main = false;
};

struct DeviceInfo {
  std::string os_name;
  std::string os_version;
  std::string cpu;
  int cpu_cores = 0;
  uint64_t ram_bytes = 0;

  std::string gpu_name;
  std::string gpu_api;            // "Metal", "Direct3D 12", "Vulkan"
  bool unified_memory = false;
  uint64_t gpu_memory_budget = 0; // bytes the GPU can comfortably use
  bool has_npu = false;           // Neural Engine / NPU present

  std::vector<std::string> hw_decoders;  // codecs with hardware decode
  std::vector<DisplayInfo> displays;

  bool low_power_mode = false;
  std::string thermal_state;      // "nominal", "fair", "serious", "critical"

  // Platform accelerators the app knows about, with availability.
  bool platform_frame_interpolation = false;  // e.g. VideoToolbox VTFrameProcessor
  std::string platform_frame_interpolation_note;
};

DeviceInfo probe_device();
std::string to_json(const DeviceInfo& info);

// Refresh rate of the main display, or 60 if unknown.
double main_display_refresh(const DeviceInfo& info);

}  // namespace advp
