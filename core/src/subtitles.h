// Subtitles: decoding embedded tracks and external files into timed cues,
// and keeping them for display. Text cues are reduced to plain lines (ASS
// styling is dropped for now; libass is a later step); bitmap cues (Blu-ray
// PGS, DVD) keep their image and position.
#pragma once

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct AVCodecContext;
struct AVPacket;
struct AVStream;

namespace advp {

struct SubtitleCue {
  double start = 0;
  double end = std::numeric_limits<double>::infinity();  // open until the next cue
  std::string text;  // plain text, lines separated by '\n'; empty for bitmaps
  // Bitmap cue: premultiplied BGRA pixels, placed at (x, y) on a canvas of
  // canvas_w x canvas_h (the video frame the track was authored for).
  std::vector<uint32_t> pixels;
  int x = 0, y = 0, w = 0, h = 0;
  int canvas_w = 0, canvas_h = 0;
};

// Plain text from an ASS dialogue line as FFmpeg's decoders produce it
// ("ReadOrder,Layer,Style,Name,MarginL,MarginR,MarginV,Effect,Text").
std::string ass_dialogue_text(const std::string& line);

// The cues of one track, safe to add from the decoder and read from the
// display at the same time. Cues survive seeks; repeats are ignored.
class SubtitleTrack {
 public:
  SubtitleTrack(std::string name, bool is_default) : name_(std::move(name)), default_(is_default) {}
  const std::string& name() const { return name_; }  // e.g. "eng", "file.srt"
  bool is_default() const { return default_; }

  void add(SubtitleCue cue);
  // Cues showing at time (usually zero or one).
  std::vector<std::shared_ptr<const SubtitleCue>> at(double time) const;

 private:
  std::string name_;
  bool default_;
  mutable std::mutex mu_;
  std::vector<std::shared_ptr<const SubtitleCue>> cues_;  // sorted by start
};

// Decodes the packets of one embedded subtitle stream into a track.
class SubtitleDecoder {
 public:
  SubtitleDecoder(AVStream* stream, SubtitleTrack& track);
  ~SubtitleDecoder();
  SubtitleDecoder(const SubtitleDecoder&) = delete;
  SubtitleDecoder& operator=(const SubtitleDecoder&) = delete;

  int stream_index() const { return index_; }
  void decode(const AVPacket* packet);

 private:
  AVCodecContext* codec_ = nullptr;
  SubtitleTrack& track_;
  int index_;
  double time_base_;
};

// Reads a whole subtitle file (.srt, .ass, .vtt, ...) into a new track.
std::unique_ptr<SubtitleTrack> load_subtitle_file(const std::string& path);

// A readable name for a stream: its language and title, if tagged.
std::string stream_label(const AVStream* stream, int number);

}  // namespace advp
