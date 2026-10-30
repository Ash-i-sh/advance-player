#include "apple/subtitle_overlay.h"

#import <AppKit/AppKit.h>
#import <CoreText/CoreText.h>
#include <simd/simd.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace advp {
namespace {

// Text size and placement, as shares of the picture's height.
constexpr double kTextHeight = 0.055;
constexpr double kBottomMargin = 0.06;
constexpr double kMaxWidth = 0.9;  // of the picture's width

CFAttributedStringRef styled(const std::string& text, CTFontRef font, bool outline, double size) {
  const CTTextAlignment align = kCTTextAlignmentCenter;
  const CTParagraphStyleSetting setting{kCTParagraphStyleSpecifierAlignment, sizeof align, &align};
  CTParagraphStyleRef paragraph = CTParagraphStyleCreate(&setting, 1);
  NSMutableDictionary* attrs = [@{
    (id)kCTFontAttributeName : (__bridge id)font,
    (id)kCTParagraphStyleAttributeName : (__bridge id)paragraph,
    (id)kCTForegroundColorAttributeName : (__bridge id)CGColorGetConstantColor(kCGColorWhite),
  } mutableCopy];
  if (outline) {
    // Stroke only (positive width, in percent of the font size), drawn
    // under the fill so the letters keep their full shape.
    attrs[(id)kCTStrokeWidthAttributeName] = @(std::max(8.0, 240.0 / size));
    attrs[(id)kCTStrokeColorAttributeName] = (__bridge id)CGColorGetConstantColor(kCGColorBlack);
  }
  CFRelease(paragraph);
  NSString* string = [NSString stringWithUTF8String:text.c_str()] ?: @"";
  return (__bridge_retained CFAttributedStringRef)[[NSAttributedString alloc] initWithString:string
                                                                                   attributes:attrs];
}

}  // namespace

SubtitleOverlay::SubtitleOverlay(id<MTLDevice> device, id<MTLLibrary> library) : device_(device) {
  MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
  d.vertexFunction = [library newFunctionWithName:@"overlay_vertex"];
  d.fragmentFunction = [library newFunctionWithName:@"overlay_fragment"];
  MTLRenderPipelineColorAttachmentDescriptor* color = d.colorAttachments[0];
  color.pixelFormat = MTLPixelFormatBGRA8Unorm;
  color.blendingEnabled = YES;  // premultiplied alpha
  color.sourceRGBBlendFactor = MTLBlendFactorOne;
  color.sourceAlphaBlendFactor = MTLBlendFactorOne;
  color.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
  color.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
  NSError* error = nil;
  pipeline_ = [device_ newRenderPipelineStateWithDescriptor:d error:&error];
  if (!pipeline_) throw std::runtime_error("cannot build subtitle pipeline");
}

void SubtitleOverlay::set(const std::vector<std::shared_ptr<const SubtitleCue>>& cues, CGRect video,
                          int frame_w, int frame_h) {
  if (cues == shown_ && CGRectEqualToRect(video, shown_video_)) return;
  shown_ = cues;
  shown_video_ = video;
  sprites_.clear();
  std::string text;
  for (const auto& cue : cues) {
    if (!cue->text.empty()) {
      text += (text.empty() ? "" : "\n") + cue->text;
    } else if (!cue->pixels.empty()) {
      add_bitmap(*cue, video, frame_w, frame_h);
    }
  }
  if (!text.empty()) add_text(text, video);
}

void SubtitleOverlay::add_text(const std::string& text, CGRect video) {
  const double size = std::max(14.0, video.size.height * kTextHeight);
  CTFontRef font = (__bridge CTFontRef)[NSFont systemFontOfSize:size weight:NSFontWeightSemibold];
  CFAttributedStringRef fill = styled(text, font, false, size);
  CFAttributedStringRef stroke = styled(text, font, true, size);
  CTFramesetterRef fill_setter = CTFramesetterCreateWithAttributedString(fill);
  CTFramesetterRef stroke_setter = CTFramesetterCreateWithAttributedString(stroke);
  const CGSize fit = CTFramesetterSuggestFrameSizeWithConstraints(
      fill_setter, CFRangeMake(0, 0), nullptr,
      CGSizeMake(video.size.width * kMaxWidth, CGFLOAT_MAX), nullptr);

  const double pad = std::ceil(size * 0.2);
  const int w = static_cast<int>(std::ceil(fit.width + 2 * pad));
  const int h = static_cast<int>(std::ceil(fit.height + 2 * pad));
  std::vector<uint32_t> pixels(static_cast<size_t>(w) * h, 0);
  CGColorSpaceRef space = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
  CGContextRef ctx = CGBitmapContextCreate(
      pixels.data(), w, h, 8, static_cast<size_t>(w) * 4, space,
      static_cast<uint32_t>(kCGImageAlphaPremultipliedFirst) | static_cast<uint32_t>(kCGBitmapByteOrder32Little));
  CGColorSpaceRelease(space);
  CGMutablePathRef path = CGPathCreateMutable();
  CGPathAddRect(path, nullptr, CGRectMake(pad, pad, std::ceil(fit.width) + 1, std::ceil(fit.height) + 1));
  // Soft shadow under the outline, then the white letters on top.
  CGContextSetShadowWithColor(ctx, CGSizeMake(0, -size * 0.04), size * 0.08,
                              CGColorGetConstantColor(kCGColorBlack));
  CTFrameRef stroke_frame = CTFramesetterCreateFrame(stroke_setter, CFRangeMake(0, 0), path, nullptr);
  CTFrameDraw(stroke_frame, ctx);
  CGContextSetShadowWithColor(ctx, CGSizeZero, 0, nullptr);
  CTFrameRef fill_frame = CTFramesetterCreateFrame(fill_setter, CFRangeMake(0, 0), path, nullptr);
  CTFrameDraw(fill_frame, ctx);
  CFRelease(stroke_frame);
  CFRelease(fill_frame);
  CGPathRelease(path);
  CGContextRelease(ctx);
  CFRelease(fill_setter);
  CFRelease(stroke_setter);
  CFRelease(fill);
  CFRelease(stroke);

  const double x = video.origin.x + (video.size.width - w) / 2;
  const double y = video.origin.y + video.size.height * (1.0 - kBottomMargin) - h + pad;
  sprites_.push_back({upload(pixels.data(), w, h, static_cast<size_t>(w) * 4), CGRectMake(x, y, w, h)});
}

void SubtitleOverlay::add_bitmap(const SubtitleCue& cue, CGRect video, int frame_w, int frame_h) {
  const double canvas_w = cue.canvas_w > 0 ? cue.canvas_w : frame_w;
  const double canvas_h = cue.canvas_h > 0 ? cue.canvas_h : frame_h;
  if (canvas_w <= 0 || canvas_h <= 0) return;
  const double sx = video.size.width / canvas_w;
  const double sy = video.size.height / canvas_h;
  const CGRect rect = CGRectMake(video.origin.x + cue.x * sx, video.origin.y + cue.y * sy,
                                 cue.w * sx, cue.h * sy);
  sprites_.push_back({upload(cue.pixels.data(), cue.w, cue.h, static_cast<size_t>(cue.w) * 4), rect});
}

id<MTLTexture> SubtitleOverlay::upload(const void* bgra, int w, int h, size_t stride) {
  MTLTextureDescriptor* d =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                         width:w
                                                        height:h
                                                     mipmapped:NO];
  d.usage = MTLTextureUsageShaderRead;
  id<MTLTexture> texture = [device_ newTextureWithDescriptor:d];
  [texture replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0 withBytes:bgra bytesPerRow:stride];
  return texture;
}

void SubtitleOverlay::encode(id<MTLRenderCommandEncoder> encoder, double target_w,
                             double target_h) const {
  if (sprites_.empty()) return;
  [encoder setViewport:MTLViewport{0, 0, target_w, target_h, 0, 1}];
  [encoder setRenderPipelineState:pipeline_];
  for (const Sprite& s : sprites_) {
    // Corners in normalised device coordinates (y up).
    const simd_float4 ndc = simd_make_float4(
        static_cast<float>(2 * s.rect.origin.x / target_w - 1),
        static_cast<float>(1 - 2 * s.rect.origin.y / target_h),
        static_cast<float>(2 * CGRectGetMaxX(s.rect) / target_w - 1),
        static_cast<float>(1 - 2 * CGRectGetMaxY(s.rect) / target_h));
    [encoder setVertexBytes:&ndc length:sizeof ndc atIndex:0];
    [encoder setFragmentTexture:s.texture atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
  }
}

}  // namespace advp
