// "Open link": turns a web page address (a video site's watch page) into the
// media streams behind it, using yt-dlp, so the video plays through our own
// pipeline. yt-dlp stays a separate program, updated on its own (PLAN.md §6):
// sites change often and fixes must not wait for an app release.
#pragma once

#include <map>
#include <string>
#include <vector>

namespace advp {

struct LinkOptions {
  std::string ytdlp = "yt-dlp";  // path to the yt-dlp program
  int max_height = 2160;
  // Video codecs this machine decodes in hardware ("av1", "hevc", ...), most
  // wanted first; streams in them are preferred.
  std::vector<std::string> hardware_codecs;
};

struct LinkStreams {
  std::string title;
  std::string format;       // the site's format ids, e.g. "401+251"
  std::string video_url;
  std::string audio_url;    // empty when audio is inside the video stream
  std::map<std::string, std::string> http_headers;  // send with every request
};

// True for http(s) addresses.
bool is_link(const std::string& input);

// Runs yt-dlp and returns the chosen streams. Throws with yt-dlp's own
// message when the page cannot be resolved.
LinkStreams resolve_link(const std::string& url, const LinkOptions& options);

}  // namespace advp
