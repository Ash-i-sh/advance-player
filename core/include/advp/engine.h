// Frame-interpolation engines and the GPU backend that hosts them.
//
// Every engine implements the same Interpolator interface, so the player and
// the converter never branch on a specific engine or device. Which engine runs
// is decided from measured benchmarks (bench.h) and the runtime governor
// (governor.h). See PLAN.md §4.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace advp {

// Smoothing tiers, from cheapest to best. Numbers match PLAN.md §4.
enum class Tier {
  Off = 0,
  Blend = 1,       // frame blending
  Memc = 2,        // classic motion estimation / compensation
  Platform = 3,    // OS or vendor accelerator (VideoToolbox, NVIDIA FRUC)
  AiRealtime = 4,  // small AI model
  AiExport = 5,    // large AI model, offline only
};

const char* tier_name(Tier tier);

// A video frame in GPU-visible memory, laid out as NV12 (8-bit) or P010
// (10-bit): a full-size luma plane and a half-size interleaved chroma plane.
class GpuFrame {
 public:
  virtual ~GpuFrame() = default;
  virtual int width() const = 0;
  virtual int height() const = 0;
  virtual bool high_bit_depth() const = 0;
  // Platform handle: a CVPixelBufferRef on Apple platforms.
  virtual void* native_handle() const = 0;
};

using GpuFramePtr = std::shared_ptr<GpuFrame>;

// What the engine learned about the current source pair.
struct PairStats {
  float unmatched = 0.f;   // fraction of blocks with no good motion match
  bool scene_cut = false;  // pair spans a cut; output repeats a source frame
};

class Interpolator {
 public:
  virtual ~Interpolator() = default;
  virtual std::string name() const = 0;
  virtual Tier tier() const = 0;
  // Analyses a new source pair. Calls normally slide: (a,b), (b,c), (c,d)...
  // and engines reuse work for the shared frame.
  virtual void set_pair(const GpuFramePtr& a, const GpuFramePtr& b) = 0;
  // Writes the frame at time t (0 = a, 1 = b) into out. Blocks until the GPU
  // has finished, so out can be handed straight to an encoder or display.
  virtual void interpolate(float t, GpuFrame& out) = 0;
  // Stats for the current pair; valid after the first interpolate() call.
  virtual PairStats last_stats() const = 0;
};

struct EngineInfo {
  std::string name;
  Tier tier;
  int quality_rank;  // higher is better looking within the same tier
  std::string description;
};

class GpuBackend {
 public:
  virtual ~GpuBackend() = default;

  // The best backend for this platform, or nullptr when none is available.
  static std::unique_ptr<GpuBackend> create_default();

  virtual std::string name() const = 0;
  virtual std::vector<EngineInfo> engines() const = 0;
  virtual std::unique_ptr<Interpolator> create_engine(const std::string& name) = 0;

  // Wraps a hardware-decoded frame without copying (retains the handle).
  virtual GpuFramePtr wrap_native(void* handle) = 0;
  // A new frame from a recycling pool.
  virtual GpuFramePtr new_frame(int width, int height, bool high_bit_depth) = 0;
  // A new frame with exactly the same size, pixel format and range as like.
  virtual GpuFramePtr new_frame_like(const GpuFrame& like) = 0;
  // CPU access for software-decoded frames and tests. Strides are in bytes.
  virtual void write_planes(GpuFrame& frame, const uint8_t* y, int y_stride,
                            const uint8_t* uv, int uv_stride) = 0;
  virtual void read_planes(const GpuFrame& frame, uint8_t* y, int y_stride,
                           uint8_t* uv, int uv_stride) = 0;
};

}  // namespace advp
