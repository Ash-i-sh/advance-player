// Small RAII and error helpers shared by the FFmpeg-based pipeline.
#pragma once

#include <stdexcept>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

namespace advp {

inline std::string av_error(int code) {
  char buf[AV_ERROR_MAX_STRING_SIZE] = {};
  av_strerror(code, buf, sizeof buf);
  return buf;
}

inline void check(int code, const char* what) {
  if (code < 0) throw std::runtime_error(std::string(what) + ": " + av_error(code));
}

struct CodecContext {
  AVCodecContext* ctx = nullptr;
  CodecContext() = default;
  CodecContext(const CodecContext&) = delete;
  CodecContext& operator=(const CodecContext&) = delete;
  ~CodecContext() { avcodec_free_context(&ctx); }
};
struct PacketPtr {
  AVPacket* pkt = av_packet_alloc();
  PacketPtr() = default;
  PacketPtr(const PacketPtr&) = delete;
  PacketPtr& operator=(const PacketPtr&) = delete;
  ~PacketPtr() { av_packet_free(&pkt); }
};
struct FramePtr {
  AVFrame* frame = av_frame_alloc();
  FramePtr() = default;
  FramePtr(const FramePtr&) = delete;
  FramePtr& operator=(const FramePtr&) = delete;
  ~FramePtr() { av_frame_free(&frame); }
};

}  // namespace advp
