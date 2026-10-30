// Draws an NV12/P010 frame to the screen: YCbCr -> RGB with the frame's own
// matrix and range, scaled to fit by the viewport.
#include <metal_stdlib>
using namespace metal;

struct PresentParams {
  float4 range;   // y_black, y_scale, c_mid, c_scale (in texture units)
  float4 matrix;  // r_cr, g_cb, g_cr, b_cb
};

struct PresentVertex {
  float4 position [[position]];
  float2 uv;
};

// One triangle that covers the viewport.
vertex PresentVertex present_vertex(uint id [[vertex_id]]) {
  const float2 p = float2((id << 1) & 2, id & 2);
  PresentVertex out;
  out.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
  out.uv = float2(p.x, 1.0 - p.y);
  return out;
}

fragment float4 present_fragment(PresentVertex in [[stage_in]],
                                 texture2d<float> luma [[texture(0)]],
                                 texture2d<float> chroma [[texture(1)]],
                                 constant PresentParams& p [[buffer(0)]]) {
  constexpr sampler s(filter::linear, address::clamp_to_edge);
  const float y = (luma.sample(s, in.uv).r - p.range.x) * p.range.y;
  const float2 c = (chroma.sample(s, in.uv).rg - p.range.z) * p.range.w;
  const float3 rgb = float3(y + p.matrix.x * c.y,
                            y - p.matrix.y * c.x - p.matrix.z * c.y,
                            y + p.matrix.w * c.x);
  return float4(saturate(rgb), 1.0);
}

// Subtitle overlay: a textured rectangle (corners in normalised device
// coordinates, x0 y0 x1 y1, y up) blended with premultiplied alpha.
struct OverlayVertex {
  float4 position [[position]];
  float2 uv;
};

vertex OverlayVertex overlay_vertex(uint id [[vertex_id]], constant float4& rect [[buffer(0)]]) {
  const float2 p = float2(id & 1, id >> 1);  // triangle strip: TL, TR, BL, BR
  OverlayVertex out;
  out.position = float4(mix(rect.x, rect.z, p.x), mix(rect.y, rect.w, p.y), 0.0, 1.0);
  out.uv = p;
  return out;
}

fragment float4 overlay_fragment(OverlayVertex in [[stage_in]], texture2d<float> image [[texture(0)]]) {
  constexpr sampler s(filter::linear, address::clamp_to_edge);
  return image.sample(s, in.uv);
}
