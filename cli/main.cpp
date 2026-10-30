// advp: developer command line for the Advance Player core.
//
//   advp probe                         device capabilities as JSON
//   advp engines                       engines available on this device
//   advp bench [--size WxH] [--fps F]  measure engines, pick one for real time
//   advp convert IN OUT [options]      smooth a file to a higher frame rate
//   advp play FILE [options]           play with live smoothing, report pacing

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "advp/bench.h"
#include "advp/convert.h"
#include "advp/device.h"
#include "advp/engine.h"
#include "advp/link.h"
#include "advp/player.h"
#include "advp/rate.h"

namespace {

void usage() {
  std::fprintf(stderr,
               "usage:\n"
               "  advp probe\n"
               "  advp engines\n"
               "  advp bench [--size 1920x1080] [--source-fps 23.976] [--fps auto|60|120]\n"
               "  advp convert IN OUT [--fps 60] [--engine auto|NAME] [--codec hevc|h264]\n"
               "                      [--quality 1-100] [--frames N]\n"
               "  advp play FILE|LINK [--fps display|fit:60|60] [--engine auto|off|NAME] [--start S]\n"
               "                 [--seconds N] [--fullscreen] [--volume 0-100] [--no-audio]\n"
               "                 [--screenshot FILE.png] [--actions 2:pause,4:pause,5:forward]\n"
               "                 [--max-height 2160]   (LINK: a video page, opened with yt-dlp)\n");
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

// yt-dlp: $ADVP_YTDLP, else the copy in the project's .venv next to the
// build directory, else the one on PATH.
std::string find_ytdlp() {
  if (const char* env = std::getenv("ADVP_YTDLP")) return env;
#if defined(__APPLE__)
  char exe[4096];
  uint32_t size = sizeof exe;
  if (_NSGetExecutablePath(exe, &size) == 0) {
    std::error_code ec;
    const auto venv = std::filesystem::canonical(exe, ec).parent_path().parent_path() / ".venv/bin/yt-dlp";
    if (!ec && std::filesystem::exists(venv)) return venv.string();
  }
#endif
  return "yt-dlp";
}

// Fills the play options from a web page address.
bool open_link(advp::PlayOptions& opt, const std::string& url, const std::vector<std::string>& args) {
  advp::LinkOptions link;
  link.ytdlp = find_ytdlp();
  if (const char* v = flag(args, "--max-height")) link.max_height = std::atoi(v);
  // Hardware-decoded codecs first; among them, the most compact first.
  const auto hw = advp::probe_device().hw_decoders;
  for (const char* codec : {"av1", "hevc", "vp9", "h264"}) {
    if (std::find(hw.begin(), hw.end(), codec) != hw.end()) link.hardware_codecs.push_back(codec);
  }
  std::fprintf(stderr, "resolving link with %s ...\n", link.ytdlp.c_str());
  try {
    const advp::LinkStreams s = advp::resolve_link(url, link);
    std::printf("link: %s (formats %s)\n", s.title.c_str(), s.format.c_str());
    opt.input = s.video_url;
    opt.audio_input = s.audio_url;
    opt.title = s.title;
    std::string headers;
    for (const auto& [key, value] : s.http_headers) headers += key + ": " + value + "\r\n";
    opt.input_options = {{"reconnect", "1"},
                         {"reconnect_on_network_error", "1"},
                         {"reconnect_delay_max", "4"}};
    if (!headers.empty()) opt.input_options["headers"] = headers;
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return false;
  }
}

int cmd_play(const std::vector<std::string>& args) {
  if (args.empty()) {
    usage();
    return 2;
  }
  advp::PlayOptions opt;
  opt.input = args[0];
  if (advp::is_link(args[0]) && !open_link(opt, args[0], args)) return 1;
  if (const char* v = flag(args, "--engine")) opt.engine = v;
  if (const char* v = flag(args, "--fps")) opt.fps = v;
  if (const char* v = flag(args, "--start")) opt.start = std::atof(v);
  if (const char* v = flag(args, "--seconds")) opt.max_seconds = std::atof(v);
  if (const char* v = flag(args, "--screenshot")) opt.screenshot = v;
  if (const char* v = flag(args, "--volume")) opt.volume = static_cast<float>(std::atof(v) / 100.0);
  if (const char* v = flag(args, "--actions")) {
    // "2:pause,4:pause,5:forward": seconds after start, then the control.
    std::string list = v;
    for (size_t pos = 0; pos < list.size();) {
      const size_t end = std::min(list.find(',', pos), list.size());
      const std::string item = list.substr(pos, end - pos);
      const size_t colon = item.find(':');
      if (colon == std::string::npos) {
        std::fprintf(stderr, "bad --actions item: %s\n", item.c_str());
        return 2;
      }
      opt.actions.push_back({std::atof(item.substr(0, colon).c_str()), item.substr(colon + 1)});
      pos = end + 1;
    }
  }
  for (const auto& a : args) {
    if (a == "--fullscreen") opt.fullscreen = true;
    if (a == "--no-audio") opt.no_audio = true;
  }

  auto backend = backend_or_die();
  const auto r = advp::play_file(*backend, opt);
  if (!r.ok) {
    std::fprintf(stderr, "play failed: %s\n", r.error.c_str());
    return 1;
  }
  std::printf(
      "%dx%d %s decode, %.3f fps -> %.3f fps on a %.1f Hz display\n"
      "engine %s -> %s (%d changes)\n"
      "%lld refreshes, %lld frames presented (%lld smoothed), %lld missed, %lld decoder underflows\n"
      "smoothing GPU time per output frame: mean %.2f ms, p99 %.2f ms, max %.2f ms\n"
      "frame timing error (RMS): %.3f ms\n",
      r.width, r.height, r.decoder.c_str(), r.source_fps, r.output_fps, r.display_hz,
      r.start_engine.c_str(),
      r.final_engine.c_str(), r.engine_changes, static_cast<long long>(r.refreshes),
      static_cast<long long>(r.presented), static_cast<long long>(r.interpolated),
      static_cast<long long>(r.missed), static_cast<long long>(r.underflows), r.work_ms_mean,
      r.work_ms_p99, r.work_ms_max, r.interval_jitter_ms);
  std::printf("audio: %s\n", r.audio.c_str());
  if (r.audio_latency_ms > 0) {
    std::printf(
        "audio sync vs. video clock: mean %.2f ms, max %.2f ms; device latency %.1f ms;\n"
        "%lld underruns, %lld resyncs\n",
        r.audio_sync_ms_mean, r.audio_sync_ms_max, r.audio_latency_ms,
        static_cast<long long>(r.audio_underruns), static_cast<long long>(r.audio_resyncs));
  }
  if (!r.even_cadence) {
    std::printf(
        "note: %.2f fps does not divide %.0f Hz evenly, so frames are held for uneven\n"
        "      numbers of refreshes (mild judder). %.2f fps would be steady: --fps fit:%.0f\n",
        r.output_fps, r.display_hz, r.even_fps, std::round(r.output_fps));
  }
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
  if (command == "play") return cmd_play(args);
  usage();
  return 2;
}
