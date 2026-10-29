#include "advp/device.h"
#include "advp/engine.h"
#include "json.h"

namespace advp {

const char* tier_name(Tier tier) {
  switch (tier) {
    case Tier::Off: return "L0 off";
    case Tier::Blend: return "L1 blend";
    case Tier::Memc: return "L2 motion";
    case Tier::Platform: return "L3 platform";
    case Tier::AiRealtime: return "L4 AI";
    case Tier::AiExport: return "L5 AI export";
  }
  return "unknown";
}

double main_display_refresh(const DeviceInfo& info) {
  for (const auto& d : info.displays) {
    if (d.main && d.refresh_hz > 0) return d.refresh_hz;
  }
  if (!info.displays.empty() && info.displays.front().refresh_hz > 0) {
    return info.displays.front().refresh_hz;
  }
  return 60.0;
}

std::string to_json(const DeviceInfo& info) {
  JsonWriter w;
  w.begin_object();
  w.field("os", info.os_name + " " + info.os_version);
  w.field("cpu", info.cpu);
  w.field("cpu_cores", info.cpu_cores);
  w.field("ram_gb", static_cast<double>(info.ram_bytes) / (1ull << 30));
  w.field("gpu", info.gpu_name);
  w.field("gpu_api", info.gpu_api);
  w.field("unified_memory", info.unified_memory);
  w.field("gpu_memory_budget_gb", static_cast<double>(info.gpu_memory_budget) / (1ull << 30));
  w.field("npu", info.has_npu);
  w.key("hw_decoders").begin_array();
  for (const auto& codec : info.hw_decoders) w.value(codec);
  w.end_array();
  w.key("displays").begin_array();
  for (const auto& d : info.displays) {
    w.begin_object();
    w.field("name", d.name);
    w.field("resolution", std::to_string(d.width) + "x" + std::to_string(d.height));
    w.field("refresh_hz", d.refresh_hz);
    w.field("hdr", d.hdr);
    w.field("main", d.main);
    w.end_object();
  }
  w.end_array();
  w.field("low_power_mode", info.low_power_mode);
  w.field("thermal_state", info.thermal_state);
  w.field("platform_frame_interpolation", info.platform_frame_interpolation);
  w.field("platform_frame_interpolation_note", info.platform_frame_interpolation_note);
  w.end_object();
  return w.str();
}

}  // namespace advp
