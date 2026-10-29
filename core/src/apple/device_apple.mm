#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <VideoToolbox/VideoToolbox.h>

#include <sys/sysctl.h>

#include "advp/device.h"

namespace advp {
namespace {

std::string sysctl_string(const char* name) {
  size_t size = 0;
  if (sysctlbyname(name, nullptr, &size, nullptr, 0) != 0 || size == 0) return {};
  std::string value(size, '\0');
  if (sysctlbyname(name, value.data(), &size, nullptr, 0) != 0) return {};
  value.resize(size > 0 ? size - 1 : 0);  // drop the trailing NUL
  return value;
}

template <typename T>
T sysctl_value(const char* name) {
  T value{};
  size_t size = sizeof value;
  sysctlbyname(name, &value, &size, nullptr, 0);
  return value;
}

std::string gpu_family(id<MTLDevice> device) {
  struct Family {
    MTLGPUFamily family;
    const char* name;
  };
  // Newest first; the first match is the device's family.
  const Family families[] = {
      {MTLGPUFamilyApple9, "Apple9"}, {MTLGPUFamilyApple8, "Apple8"},
      {MTLGPUFamilyApple7, "Apple7"}, {MTLGPUFamilyApple6, "Apple6"},
      {MTLGPUFamilyMac2, "Mac2"},
  };
  for (const auto& f : families) {
    if ([device supportsFamily:f.family]) return f.name;
  }
  return "unknown";
}

void probe_decoders(DeviceInfo& info) {
  struct Codec {
    CMVideoCodecType type;
    const char* name;
  };
  // FourCCs rather than SDK constants so older SDKs still build.
  const Codec codecs[] = {
      {'avc1', "h264"}, {'hvc1', "hevc"}, {'vp09', "vp9"},
      {'av01', "av1"},  {'apcn', "prores"}, {'mp2v', "mpeg2"},
  };
  for (const auto& c : codecs) {
    if (VTIsHardwareDecodeSupported(c.type)) info.hw_decoders.push_back(c.name);
  }
}

void probe_displays(DeviceInfo& info) {
  NSScreen* main = [NSScreen mainScreen];
  for (NSScreen* screen in [NSScreen screens]) {
    DisplayInfo d;
    d.name = screen.localizedName.UTF8String ?: "";
    const CGFloat scale = screen.backingScaleFactor;
    d.width = static_cast<int>(screen.frame.size.width * scale);
    d.height = static_cast<int>(screen.frame.size.height * scale);
    d.refresh_hz = static_cast<double>(screen.maximumFramesPerSecond);
    d.hdr = screen.maximumPotentialExtendedDynamicRangeColorComponentValue > 1.0;
    d.main = screen == main;
    info.displays.push_back(d);
  }
}

void probe_platform_interpolation(DeviceInfo& info) {
  // VideoToolbox frame-rate conversion (VTFrameProcessor) ships in macOS
  // 15.4. Building against it also needs the macOS 15.4+ SDK (Xcode 16.3+).
  const NSOperatingSystemVersion v = [NSProcessInfo processInfo].operatingSystemVersion;
  const bool os_ok = v.majorVersion > 15 || (v.majorVersion == 15 && v.minorVersion >= 4);
  const bool class_present = NSClassFromString(@"VTFrameProcessor") != nil;
  info.platform_frame_interpolation = false;  // no engine wired up yet
  if (!os_ok) {
    info.platform_frame_interpolation_note =
        "VideoToolbox frame-rate conversion needs macOS 15.4 or later";
  } else if (!class_present) {
    info.platform_frame_interpolation_note = "VideoToolbox frame processor not found";
  } else {
    info.platform_frame_interpolation_note =
        "available on this OS; engine needs the macOS 15.4+ SDK to build";
  }
}

const char* thermal_name(NSProcessInfoThermalState state) {
  switch (state) {
    case NSProcessInfoThermalStateNominal: return "nominal";
    case NSProcessInfoThermalStateFair: return "fair";
    case NSProcessInfoThermalStateSerious: return "serious";
    case NSProcessInfoThermalStateCritical: return "critical";
  }
  return "unknown";
}

}  // namespace

DeviceInfo probe_device() {
  @autoreleasepool {
    DeviceInfo info;
    NSProcessInfo* process = [NSProcessInfo processInfo];
    const NSOperatingSystemVersion v = process.operatingSystemVersion;
    info.os_name = "macOS";
    info.os_version = std::to_string(v.majorVersion) + "." + std::to_string(v.minorVersion) +
                      "." + std::to_string(v.patchVersion);
    info.cpu = sysctl_string("machdep.cpu.brand_string");
    info.cpu_cores = sysctl_value<int>("hw.ncpu");
    info.ram_bytes = sysctl_value<uint64_t>("hw.memsize");

    if (id<MTLDevice> device = MTLCreateSystemDefaultDevice()) {
      info.gpu_name = device.name.UTF8String;
      info.gpu_api = "Metal (" + gpu_family(device) + ")";
      info.unified_memory = device.hasUnifiedMemory;
      info.gpu_memory_budget = device.recommendedMaxWorkingSetSize;
    }
#if defined(__aarch64__)
    info.has_npu = true;  // every Apple Silicon Mac has a Neural Engine
#endif

    probe_decoders(info);
    probe_displays(info);
    probe_platform_interpolation(info);
    info.low_power_mode = process.lowPowerModeEnabled;
    info.thermal_state = thermal_name(process.thermalState);
    return info;
  }
}

}  // namespace advp
