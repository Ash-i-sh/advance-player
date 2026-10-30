#include "subtitles.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "ffmpeg_util.h"

namespace advp {
namespace {

// Premultiplied BGRA from a palette entry (0xAARRGGBB).
uint32_t premultiply(uint32_t argb) {
  const uint32_t a = argb >> 24;
  const uint32_t r = ((argb >> 16) & 0xff) * a / 255;
  const uint32_t g = ((argb >> 8) & 0xff) * a / 255;
  const uint32_t b = (argb & 0xff) * a / 255;
  return (a << 24) | (r << 16) | (g << 8) | b;  // little-endian memory: B, G, R, A
}

std::vector<SubtitleCue> cues_from(const AVSubtitle& sub, double packet_time, double duration,
                                   int canvas_w, int canvas_h) {
  const double start = packet_time + sub.start_display_time / 1000.0;
  double end = std::numeric_limits<double>::infinity();
  if (sub.end_display_time != UINT32_MAX && sub.end_display_time > sub.start_display_time) {
    end = packet_time + sub.end_display_time / 1000.0;
  } else if (duration > 0) {
    end = start + duration;
  }
  std::vector<SubtitleCue> cues;
  std::string text;
  for (unsigned i = 0; i < sub.num_rects; ++i) {
    const AVSubtitleRect* r = sub.rects[i];
    if (r->type == SUBTITLE_ASS && r->ass) {
      const std::string line = ass_dialogue_text(r->ass);
      if (!line.empty()) text += (text.empty() ? "" : "\n") + line;
    } else if (r->type == SUBTITLE_TEXT && r->text) {
      text += (text.empty() ? "" : "\n") + std::string(r->text);
    } else if (r->type == SUBTITLE_BITMAP && r->w > 0 && r->h > 0 && r->data[0] && r->data[1]) {
      SubtitleCue cue;
      cue.start = start;
      cue.end = end;
      cue.x = r->x;
      cue.y = r->y;
      cue.w = r->w;
      cue.h = r->h;
      cue.canvas_w = canvas_w;
      cue.canvas_h = canvas_h;
      const auto* palette = reinterpret_cast<const uint32_t*>(r->data[1]);
      cue.pixels.resize(static_cast<size_t>(r->w) * r->h);
      for (int y = 0; y < r->h; ++y) {
        const uint8_t* row = r->data[0] + static_cast<size_t>(y) * r->linesize[0];
        for (int x = 0; x < r->w; ++x) {
          cue.pixels[static_cast<size_t>(y) * r->w + x] = premultiply(palette[row[x]]);
        }
      }
      cues.push_back(std::move(cue));
    }
  }
  if (!text.empty()) {
    SubtitleCue cue;
    cue.start = start;
    cue.end = end;
    cue.text = std::move(text);
    cues.push_back(std::move(cue));
  }
  if (cues.empty()) {
    // An empty event clears the screen (how bitmap tracks end their cues).
    SubtitleCue clear;
    clear.start = start;
    clear.end = start;
    cues.push_back(std::move(clear));
  }
  return cues;
}

}  // namespace

std::string ass_dialogue_text(const std::string& line) {
  // Text is the ninth field and may itself contain commas.
  size_t at = 0;
  for (int field = 0; field < 8; ++field) {
    at = line.find(',', at);
    if (at == std::string::npos) return {};
    ++at;
  }
  std::string text;
  for (size_t i = at; i < line.size(); ++i) {
    const char c = line[i];
    if (c == '{') {  // override tags: {\i1}, {\pos(...)} ...
      const size_t close = line.find('}', i);
      if (close == std::string::npos) break;
      i = close;
    } else if (c == '\\' && i + 1 < line.size() && (line[i + 1] == 'N' || line[i + 1] == 'n')) {
      text += '\n';
      ++i;
    } else if (c == '\\' && i + 1 < line.size() && line[i + 1] == 'h') {
      text += ' ';
      ++i;
    } else if (c != '\r') {
      text += c;
    }
  }
  while (!text.empty() && (text.back() == '\n' || text.back() == ' ')) text.pop_back();
  return text;
}

void SubtitleTrack::add(SubtitleCue cue) {
  std::lock_guard<std::mutex> lock(mu_);
  for (auto& c : cues_) {
    // Already have it (read again after a seek back).
    if (std::fabs(c->start - cue.start) < 0.001 && c->text == cue.text && c->w == cue.w &&
        c->h == cue.h && c->x == cue.x && c->y == cue.y) {
      return;
    }
    // A cue left open ends where the next one starts.
    if (std::isinf(c->end) && c->start < cue.start) {
      auto closed = std::make_shared<SubtitleCue>(*c);
      closed->end = cue.start;
      c = std::move(closed);
    }
  }
  if (cue.end <= cue.start) return;  // a clear event: only closes what is open
  auto it = std::upper_bound(cues_.begin(), cues_.end(), cue.start,
                             [](double t, const auto& c) { return t < c->start; });
  cues_.insert(it, std::make_shared<const SubtitleCue>(std::move(cue)));
}

std::vector<std::shared_ptr<const SubtitleCue>> SubtitleTrack::at(double time) const {
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<std::shared_ptr<const SubtitleCue>> out;
  auto end = std::upper_bound(cues_.begin(), cues_.end(), time,
                              [](double t, const auto& c) { return t < c->start; });
  for (auto it = cues_.begin(); it != end; ++it) {
    if ((*it)->end > time) out.push_back(*it);
  }
  return out;
}

SubtitleDecoder::SubtitleDecoder(AVStream* stream, SubtitleTrack& track)
    : track_(track), index_(stream->index), time_base_(av_q2d(stream->time_base)) {
  const AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);
  if (!codec) throw std::runtime_error("no decoder for this subtitle format");
  codec_ = avcodec_alloc_context3(codec);
  try {
    check(avcodec_parameters_to_context(codec_, stream->codecpar), "subtitle parameters");
    codec_->pkt_timebase = stream->time_base;
    check(avcodec_open2(codec_, codec, nullptr), "open subtitle decoder");
  } catch (...) {
    avcodec_free_context(&codec_);
    throw;
  }
}

SubtitleDecoder::~SubtitleDecoder() { avcodec_free_context(&codec_); }

void SubtitleDecoder::decode(const AVPacket* packet) {
  AVSubtitle sub{};
  int got = 0;
  // A damaged packet costs one line of text, not playback.
  if (avcodec_decode_subtitle2(codec_, &sub, &got, packet) < 0 || !got) return;
  const double time = packet->pts != AV_NOPTS_VALUE ? packet->pts * time_base_ : 0.0;
  const double duration = packet->duration > 0 ? packet->duration * time_base_ : 0.0;
  for (auto& cue : cues_from(sub, time, duration, codec_->width, codec_->height)) {
    track_.add(std::move(cue));
  }
  avsubtitle_free(&sub);
}

std::unique_ptr<SubtitleTrack> load_subtitle_file(const std::string& path) {
  AVFormatContext* format = nullptr;
  check(avformat_open_input(&format, path.c_str(), nullptr, nullptr), "open subtitle file");
  std::unique_ptr<SubtitleTrack> track;
  try {
    check(avformat_find_stream_info(format, nullptr), "read subtitle file");
    const int index = av_find_best_stream(format, AVMEDIA_TYPE_SUBTITLE, -1, -1, nullptr, 0);
    check(index, "find subtitles in file");
    const size_t slash = path.find_last_of('/');
    track = std::make_unique<SubtitleTrack>(slash == std::string::npos ? path : path.substr(slash + 1),
                                            true);
    SubtitleDecoder decoder(format->streams[index], *track);
    PacketPtr packet;
    while (av_read_frame(format, packet.pkt) >= 0) {
      if (packet.pkt->stream_index == index) decoder.decode(packet.pkt);
      av_packet_unref(packet.pkt);
    }
  } catch (...) {
    avformat_close_input(&format);
    throw;
  }
  avformat_close_input(&format);
  return track;
}

std::string stream_label(const AVStream* stream, int number) {
  std::string label = "#" + std::to_string(number);
  const AVDictionaryEntry* lang = av_dict_get(stream->metadata, "language", nullptr, 0);
  const AVDictionaryEntry* title = av_dict_get(stream->metadata, "title", nullptr, 0);
  if (lang && lang->value[0]) label += std::string(" ") + lang->value;
  if (title && title->value[0]) label += std::string(" \"") + title->value + "\"";
  return label;
}

}  // namespace advp
