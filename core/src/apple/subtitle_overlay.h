// Draws subtitle cues over the picture after smoothing, so text is never
// interpolated or warped. Each set of cues is turned into textures once,
// when it appears; every frame then only blends those textures.
#pragma once

#import <CoreGraphics/CoreGraphics.h>
#import <Metal/Metal.h>

#include <memory>
#include <vector>

#include "subtitles.h"

namespace advp {

class SubtitleOverlay {
 public:
  // library: holds overlay_vertex / overlay_fragment (present.metal).
  SubtitleOverlay(id<MTLDevice> device, id<MTLLibrary> library);

  // Cues to show, for a picture occupying video (drawable pixels, origin top
  // left) whose frames are frame_w x frame_h. Rebuilds textures only when
  // the cues or the picture size change.
  void set(const std::vector<std::shared_ptr<const SubtitleCue>>& cues, CGRect video, int frame_w,
           int frame_h);

  const std::vector<std::shared_ptr<const SubtitleCue>>& cues() const { return shown_; }

  // Blends the current cues into an encoder whose target is target_w x target_h.
  void encode(id<MTLRenderCommandEncoder> encoder, double target_w, double target_h) const;

 private:
  struct Sprite {
    id<MTLTexture> texture;
    CGRect rect;  // drawable pixels, origin top left
  };
  void add_text(const std::string& text, CGRect video);
  void add_bitmap(const SubtitleCue& cue, CGRect video, int frame_w, int frame_h);
  id<MTLTexture> upload(const void* bgra, int w, int h, size_t stride);

  id<MTLDevice> device_;
  id<MTLRenderPipelineState> pipeline_;
  std::vector<std::shared_ptr<const SubtitleCue>> shown_;
  CGRect shown_video_ = CGRectZero;
  std::vector<Sprite> sprites_;
};

}  // namespace advp
