#include "advp/link.h"

#include <poll.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <fcntl.h>

#include <cstring>
#include <sstream>
#include <stdexcept>

extern char** environ;

namespace advp {
namespace {

struct ProcessOutput {
  int status = -1;
  std::string out;
  std::string err;
};

// Runs a program with these arguments, no shell involved (a pasted address
// can never become a command), and collects what it prints.
ProcessOutput run(const std::vector<std::string>& args) {
  int out_pipe[2];
  int err_pipe[2];
  if (pipe(out_pipe) != 0) throw std::runtime_error("cannot create pipe");
  if (pipe(err_pipe) != 0) {
    close(out_pipe[0]);
    close(out_pipe[1]);
    throw std::runtime_error("cannot create pipe");
  }
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_adddup2(&actions, out_pipe[1], STDOUT_FILENO);
  posix_spawn_file_actions_adddup2(&actions, err_pipe[1], STDERR_FILENO);
  posix_spawn_file_actions_addclose(&actions, out_pipe[0]);
  posix_spawn_file_actions_addclose(&actions, err_pipe[0]);
  posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);

  std::vector<char*> argv;
  for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
  argv.push_back(nullptr);
  pid_t pid = 0;
  const int rc = posix_spawnp(&pid, argv[0], &actions, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  close(out_pipe[1]);
  close(err_pipe[1]);
  if (rc != 0) {
    close(out_pipe[0]);
    close(err_pipe[0]);
    throw std::runtime_error("cannot run " + args[0] + ": " + std::strerror(rc));
  }

  ProcessOutput result;
  pollfd fds[2] = {{out_pipe[0], POLLIN, 0}, {err_pipe[0], POLLIN, 0}};
  std::string* sinks[2] = {&result.out, &result.err};
  int open_fds = 2;
  char buf[4096];
  while (open_fds > 0) {
    if (poll(fds, 2, -1) < 0) break;
    for (int i = 0; i < 2; ++i) {
      if (fds[i].fd < 0 || !(fds[i].revents & (POLLIN | POLLHUP))) continue;
      const ssize_t n = read(fds[i].fd, buf, sizeof buf);
      if (n > 0) {
        sinks[i]->append(buf, static_cast<size_t>(n));
      } else {
        close(fds[i].fd);
        fds[i].fd = -1;
        --open_fds;
      }
    }
  }
  waitpid(pid, &result.status, 0);
  return result;
}

// yt-dlp format filter for one codec family.
std::string codec_filter(const std::string& codec) {
  if (codec == "av1") return "[vcodec^=av01]";
  if (codec == "vp9") return "[vcodec~='^vp0?9']";
  if (codec == "hevc") return "[vcodec~='^(hev1|hvc1|hevc)']";
  if (codec == "h264") return "[vcodec^=avc1]";
  return "";
}

// Separate video and audio over plain HTTPS (seekable, no playlist
// round-trips), video in a hardware-decoded codec where possible; failing
// that, whatever the site has.
std::string format_selector(const LinkOptions& options) {
  const std::string height = "[height<=" + std::to_string(options.max_height) + "]";
  std::string video;
  for (const auto& codec : options.hardware_codecs) {
    const std::string filter = codec_filter(codec);
    if (!filter.empty()) video += "bv*" + height + filter + "[protocol=https]/";
  }
  video += "bv*" + height + "[protocol=https]";
  // "-drc" tracks have their loudness range squeezed for phones.
  const std::string audio = "ba[acodec=opus][format_id!$=-drc]/ba[format_id!$=-drc]/ba";
  return "(" + video + ")+(" + audio + ")/b" + height + "/b";
}

// Reads {"key": "value", ...} with string values (yt-dlp's http_headers).
std::map<std::string, std::string> parse_string_map(const std::string& json) {
  std::map<std::string, std::string> map;
  size_t i = 0;
  const auto read_string = [&](std::string& s) {
    s.clear();
    for (++i; i < json.size() && json[i] != '"'; ++i) {
      if (json[i] == '\\' && i + 1 < json.size()) {
        ++i;
        switch (json[i]) {
          case 'n': s += '\n'; break;
          case 't': s += '\t'; break;
          case 'u': i += 4; s += '?'; break;  // not used in header values
          default: s += json[i];
        }
      } else {
        s += json[i];
      }
    }
    ++i;
  };
  std::string key;
  std::string value;
  while ((i = json.find('"', i)) != std::string::npos) {
    read_string(key);
    i = json.find('"', i);
    if (i == std::string::npos) break;
    read_string(value);
    map[key] = value;
  }
  return map;
}

}  // namespace

bool is_link(const std::string& input) {
  return input.rfind("http://", 0) == 0 || input.rfind("https://", 0) == 0;
}

LinkStreams resolve_link(const std::string& url, const LinkOptions& options) {
  const ProcessOutput p = run({
      options.ytdlp, "--no-playlist", "--no-warnings", "--js-runtimes", "node",
      "-f", format_selector(options),
      "--print", "%(title)s", "--print", "%(format_id)s", "--print",
      "%(http_headers,requested_formats.0.http_headers)j",  // per stream when split
      "--print", "urls",
      "--", url,  // never read as an option
  });
  if (!WIFEXITED(p.status) || WEXITSTATUS(p.status) != 0) {
    std::string message = p.err;
    const size_t at = message.rfind("ERROR: ");
    if (at != std::string::npos) message = message.substr(at + 7);
    while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) message.pop_back();
    throw std::runtime_error("cannot open link: " + (message.empty() ? "yt-dlp failed" : message));
  }
  std::istringstream lines(p.out);
  std::string headers;
  LinkStreams s;
  std::getline(lines, s.title);
  std::getline(lines, s.format);
  std::getline(lines, headers);
  std::getline(lines, s.video_url);
  std::getline(lines, s.audio_url);
  if (s.video_url.empty()) throw std::runtime_error("cannot open link: no stream found");
  s.http_headers = parse_string_map(headers);
  return s;
}

}  // namespace advp
