// advp: developer command line for the Advance Player core.
//
//   advp probe                         device capabilities as JSON
//   advp engines                       engines available on this device
//   advp bench [--size WxH] [--fps F]  measure engines, pick one for real time
//   advp convert IN OUT [options]      smooth a file to a higher frame rate

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "advp/bench.h"
#include "advp/convert.h"
#include "advp/device.h"
#include "advp/engine.h"
#include "advp/rate.h"

namespace {

void usage() {
  std::fprintf(stderr,
               "usage:\n"
               "  advp probe\n"
               "  advp engines\n"
               "  advp bench [--size 1920x1080] [--source-fps 23.976] [--fps auto|60|120]\n"
               "  advp convert IN OUT [--fps 60] [--engine auto|NAME] [--codec hevc|h264]\n"
               "                      [--quality 1-100] [--frames N]\n");
}

// Returns the value after a flag, or nullptr if the flag is absent.
const char* flag(const std::vector<std::string>& args, const char* name) {
  for (size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == name) return args[i + 1].c_str();
  }
  return nullptr;
}

std::unique_ptr<advp::GpuBackend> backend_or_die() {
  auto backend = advp::GpuBackend::create_default();
  if (!backend) {
    std::fprintf(stderr, "no supported GPU found\n");
    std::exit(1);
  }
  return backend;
}

int cmd_probe() {
  std::printf("%s\n", advp::to_json(advp::probe_device()).c_str());
  return 0;
}

int cmd_engines() {
  auto backend = backend_or_die();
  std::printf("backend: %s\n", backend->name().c_str());
  for (const auto& e : backend->engines()) {
    std::printf("  %-12s %-12s %s\n", e.name.c_str(), advp::tier_name(e.tier),
                e.description.c_str());
  }
  return 0;
}

int cmd_bench(const std::vector<std::string>& args) {
  int w = 1920, h = 1080;
  if (const char* size = flag(args, "--size")) {
    if (std::sscanf(size, "%dx%d", &w, &h) != 2 || w < 64 || h < 64) {
      std::fprintf(stderr, "bad --size\n");
      return 2;
    }
  }
  const double source_fps = std::atof(flag(args, "--source-fps") ? flag(args, "--source-fps")
                                                                  : "23.976");
  const char* fps_arg = flag(args, "--fps");
  double target_fps = 0;
  if (!fps_arg || std::strcmp(fps_arg, "auto") == 0) {
    // Match the display: smoothing to anything else brings judder back.
    target_fps = advp::main_display_refresh(advp::probe_device());
  } else {
    target_fps = std::atof(fps_arg);
  }

  auto backend = backend_or_die();
  const auto results = advp::benchmark_engines(*backend, w, h);
  const auto selection = advp::select_engine(results, source_fps, target_fps);
  std::printf("%s\n", advp::to_json(results, selection, source_fps, target_fps).c_str());
  return 0;
}

int cmd_convert(const std::vector<std::string>& args) {
  if (args.size() < 2) {
    usage();
    return 2;
  }
  advp::ConvertOptions opt;
  opt.input = args[0];
  opt.output = args[1];
  if (const char* v = flag(args, "--fps")) opt.target_fps = v;
  if (const char* v = flag(args, "--engine")) opt.engine = v;
  if (const char* v = flag(args, "--codec")) opt.codec = v;
  if (const char* v = flag(args, "--quality")) opt.quality = std::atoi(v);
  if (const char* v = flag(args, "--frames")) opt.max_source_frames = std::atoll(v);
  opt.on_progress = [](const advp::ConvertProgress& p) {
    const double speed = p.elapsed > 0 ? p.media_seconds / p.elapsed : 0;
    if (p.duration > 0) {
      std::fprintf(stderr, "\r  %5.1f%%  %lld frames out  %.2fx real time   ",
                   100.0 * p.media_seconds / p.duration, static_cast<long long>(p.output_frames),
                   speed);
    } else {
      std::fprintf(stderr, "\r  %lld frames out  %.2fx real time   ",
                   static_cast<long long>(p.output_frames), speed);
    }
  };

  auto backend = backend_or_die();
  const auto r = advp::convert_file(*backend, opt);
  std::fprintf(stderr, "\n");
  if (!r.ok) {
    std::fprintf(stderr, "convert failed: %s\n", r.error.c_str());
    return 1;
  }
  const double media = r.source_fps > 0 ? r.source_frames / r.source_fps : 0;
  std::printf(
      "engine %s, %s decode, %.3f -> %.3f fps\n"
      "%lld source frames -> %lld output frames (%lld interpolated, %lld scene cuts)\n"
      "%.1f s for %.1f s of video (%.2fx real time)\n",
      r.engine.c_str(), r.decoder.c_str(), r.source_fps, r.target_fps,
      static_cast<long long>(r.source_frames), static_cast<long long>(r.output_frames),
      static_cast<long long>(r.interpolated), static_cast<long long>(r.scene_cuts), r.seconds,
      media, r.seconds > 0 ? media / r.seconds : 0.0);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 2;
  }
  const std::string command = argv[1];
  const std::vector<std::string> args(argv + 2, argv + argc);
  if (command == "probe") return cmd_probe();
  if (command == "engines") return cmd_engines();
  if (command == "bench") return cmd_bench(args);
  if (command == "convert") return cmd_convert(args);
  usage();
  return 2;
}
