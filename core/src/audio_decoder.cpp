#include "audio_decoder.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "ffmpeg_util.h"

extern "C" {
#include <libavutil/channel_layout.h>
#include <libswresample/swresample.h>
}

namespace advp {

AudioDecoder::AudioDecoder(AVStream* stream, int out_rate) : index_(stream->index) {
  frame_ = av_frame_alloc();
  try {
    const AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);
    if (!codec) throw std::runtime_error("no decoder for this audio codec");
    codec_ = avcodec_alloc_context3(codec);
    check(avcodec_parameters_to_context(codec_, stream->codecpar), "audio decoder parameters");
    codec_->pkt_timebase = stream->time_base;
    check(avcodec_open2(codec_, codec, nullptr), "open audio decoder");
    if (codec_->sample_rate <= 0) throw std::runtime_error("audio stream has no sample rate");
    rate_ = out_rate > 0 ? out_rate : codec_->sample_rate;
    time_base_ = av_q2d(stream->time_base);
  } catch (...) {
    avcodec_free_context(&codec_);
    av_frame_free(&frame_);
    throw;
  }
}

AudioDecoder::~AudioDecoder() {
  avcodec_free_context(&codec_);
  av_frame_free(&frame_);
  swr_free(&swr_);
  std::free(out_);
}

std::string AudioDecoder::description() const {
  char buf[96];
  std::snprintf(buf, sizeof buf, "%s, %d Hz, %d ch", avcodec_get_name(codec_->codec_id),
                codec_->sample_rate, codec_->ch_layout.nb_channels);
  return buf;
}

void AudioDecoder::decode(const AVPacket* packet,
                          const std::function<void(const AudioChunk&)>& sink) {
  // A damaged packet costs a moment of sound, not playback.
  if (avcodec_send_packet(codec_, packet) < 0) return;
  while (avcodec_receive_frame(codec_, frame_) >= 0) {
    convert(frame_, sink);
    av_frame_unref(frame_);
  }
}

void AudioDecoder::flush() {
  avcodec_flush_buffers(codec_);
  have_time_ = false;
  if (swr_) swr_init(swr_);  // drops samples held for resampling
}

void AudioDecoder::convert(AVFrame* frame, const std::function<void(const AudioChunk&)>& sink) {
  if (!swr_ || frame->format != swr_format_ || frame->ch_layout.nb_channels != swr_channels_) {
    swr_free(&swr_);
    const AVChannelLayout stereo = AV_CHANNEL_LAYOUT_STEREO;
    check(swr_alloc_set_opts2(&swr_, &stereo, AV_SAMPLE_FMT_FLT, rate_, &frame->ch_layout,
                              static_cast<AVSampleFormat>(frame->format), frame->sample_rate, 0,
                              nullptr),
          "audio converter");
    check(swr_init(swr_), "audio converter");
    swr_format_ = frame->format;
    swr_channels_ = frame->ch_layout.nb_channels;
  }
  const int capacity = swr_get_out_samples(swr_, frame->nb_samples);
  if (capacity > out_capacity_) {
    std::free(out_);
    out_ = static_cast<float*>(std::malloc(sizeof(float) * channels() * capacity));
    if (!out_) throw std::bad_alloc();
    out_capacity_ = capacity;
  }
  // Samples the resampler still holds from earlier frames come out first.
  const double held = static_cast<double>(swr_get_delay(swr_, rate_)) / rate_;
  uint8_t* out[1] = {reinterpret_cast<uint8_t*>(out_)};
  const int frames = swr_convert(swr_, out, capacity, frame->extended_data, frame->nb_samples);
  if (frames <= 0) return;

  int64_t ts = frame->best_effort_timestamp;
  if (ts == AV_NOPTS_VALUE) ts = frame->pts;
  const double time =
      ts != AV_NOPTS_VALUE ? ts * time_base_ - held : (have_time_ ? next_time_ : 0.0);
  next_time_ = time + static_cast<double>(frames) / rate_;
  have_time_ = true;
  sink(AudioChunk{out_, frames, time});
}

AudioSource::AudioSource(const std::string& path, const OpenOptions& options) {
  format_ = open_input(path, options);
  try {
    const int index = av_find_best_stream(format_, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    check(index, "find audio stream");
    decoder_ = std::make_unique<AudioDecoder>(format_->streams[index]);
    packet_ = av_packet_alloc();
  } catch (...) {
    avformat_close_input(&format_);
    throw;
  }
}

AudioSource::~AudioSource() {
  av_packet_free(&packet_);
  avformat_close_input(&format_);
}

bool AudioSource::read(const std::function<void(const AudioChunk&)>& sink) {
  while (true) {
    if (av_read_frame(format_, packet_) < 0) return false;
    const bool ours = packet_->stream_index == decoder_->stream_index();
    if (ours) decoder_->decode(packet_, sink);
    av_packet_unref(packet_);
    if (ours) return true;
  }
}

void AudioSource::seek(double seconds) {
  const int64_t ts = static_cast<int64_t>(seconds * AV_TIME_BASE);
  check(avformat_seek_file(format_, -1, INT64_MIN, ts, ts, 0), "seek audio");
  decoder_->flush();
}

}  // namespace advp
