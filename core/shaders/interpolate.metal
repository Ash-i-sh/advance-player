// Motion-compensated frame interpolation (tier L2) and frame blending (L1).
//
// Pipeline per source pair (a, b):
//   1. Luma pyramid for each frame (downsample2x).
//   2. Hierarchical 8x8 block matching, coarse to fine, in both directions
//      (motion_search): forward a->b and backward b->a vector fields.
//   3. 3x3 vector median to remove outliers (vector_median).
// Per output frame at time t:
//   4. interpolate_frame: each pixel follows both fields to the matching points
//      in a and b, weights the two predictions by how well they agree, and falls
//      back to one side where content is covered or uncovered (occlusion).
// A pair whose blocks mostly fail to match is a scene cut: output repeats the
// nearest source frame instead of morphing between two different shots.

#include <metal_stdlib>
using namespace metal;

constexpr sampler kLinear(coord::pixel, address::clamp_to_edge, filter::linear);

// ---------------------------------------------------------------- pyramid

kernel void downsample2x(texture2d<float, access::sample> src [[texture(0)]],
                         texture2d<float, access::write> dst [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
  // The shared corner of four source texels: bilinear gives their average.
  dst.write(float4(src.sample(kLinear, float2(gid) * 2.0f + 1.0f).r), gid);
}

// ---------------------------------------------------------- motion search

struct SearchParams {
  int2 field_size;         // blocks at this level
  int2 coarse_field_size;  // blocks at the coarser level; (0,0) at the coarsest
  int radius;              // exhaustive search radius, coarsest level only
  float lambda;            // cost per pixel of deviation from the prediction
  int refine_iters;        // integer-pel refinement rounds
  int subpel;              // 1 => half-pel refinement
  float bad_sad;           // mean abs difference (0..255) meaning "no match"
  int count_stats;         // 1 => count unmatched blocks into stats[0]
};

// Mean absolute difference (0..255) between an 8x8 block and ref displaced by v.
static float block_sad(thread const float* blk, texture2d<float, access::sample> ref,
                       float2 origin, float2 v) {
  float s = 0.0f;
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      s += fabs(blk[y * 8 + x] - ref.sample(kLinear, origin + float2(x, y) + 0.5f + v).r);
    }
  }
  return s * (255.0f / 64.0f);
}

static void try_vector(thread const float* blk, texture2d<float, access::sample> ref,
                       float2 origin, float2 v, float2 pred, float lambda,
                       thread float2& best, thread float& best_cost, thread float& best_sad) {
  const float sad = block_sad(blk, ref, origin, v);
  const float2 d = abs(v - pred);
  const float cost = sad + lambda * (d.x + d.y);
  if (cost < best_cost) {
    best_cost = cost;
    best_sad = sad;
    best = v;
  }
}

static float2 coarse_vector(texture2d<float, access::read> coarse, int2 at, int2 size) {
  return 2.0f * coarse.read(uint2(clamp(at, int2(0), size - 1))).xy;
}

kernel void motion_search(texture2d<float, access::sample> cur [[texture(0)]],
                          texture2d<float, access::sample> ref [[texture(1)]],
                          texture2d<float, access::read> coarse [[texture(2)]],
                          texture2d<float, access::write> field [[texture(3)]],
                          constant SearchParams& p [[buffer(0)]],
                          device atomic_uint* stats [[buffer(1)]],
                          uint2 gid [[thread_position_in_grid]]) {
  if (int(gid.x) >= p.field_size.x || int(gid.y) >= p.field_size.y) return;

  const float2 origin = float2(gid) * 8.0f;
  float blk[64];
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      blk[y * 8 + x] = cur.sample(kLinear, origin + float2(x, y) + 0.5f).r;
    }
  }

  float2 best = float2(0.0f);
  float best_cost = INFINITY;
  float best_sad = INFINITY;
  float2 pred = float2(0.0f);

  if (p.coarse_field_size.x == 0) {
    for (int dy = -p.radius; dy <= p.radius; ++dy) {
      for (int dx = -p.radius; dx <= p.radius; ++dx) {
        try_vector(blk, ref, origin, float2(dx, dy), pred, p.lambda, best, best_cost, best_sad);
      }
    }
  } else {
    const int2 c = int2(gid) / 2;
    const int2 cs = p.coarse_field_size;
    pred = coarse_vector(coarse, c, cs);
    try_vector(blk, ref, origin, pred, pred, p.lambda, best, best_cost, best_sad);
    try_vector(blk, ref, origin, float2(0.0f), pred, p.lambda, best, best_cost, best_sad);
    try_vector(blk, ref, origin, coarse_vector(coarse, c + int2(1, 0), cs), pred, p.lambda,
               best, best_cost, best_sad);
    try_vector(blk, ref, origin, coarse_vector(coarse, c + int2(-1, 0), cs), pred, p.lambda,
               best, best_cost, best_sad);
    try_vector(blk, ref, origin, coarse_vector(coarse, c + int2(0, 1), cs), pred, p.lambda,
               best, best_cost, best_sad);
    try_vector(blk, ref, origin, coarse_vector(coarse, c + int2(0, -1), cs), pred, p.lambda,
               best, best_cost, best_sad);
  }

  for (int i = 0; i < p.refine_iters; ++i) {
    const float2 center = best;
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0) continue;
        try_vector(blk, ref, origin, center + float2(dx, dy), pred, p.lambda,
                   best, best_cost, best_sad);
      }
    }
  }
  if (p.subpel != 0) {
    const float2 center = best;
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0) continue;
        try_vector(blk, ref, origin, center + 0.5f * float2(dx, dy), pred, p.lambda,
                   best, best_cost, best_sad);
      }
    }
  }

  field.write(float4(best, best_sad, 0.0f), gid);
  if (p.count_stats != 0 && best_sad > p.bad_sad) {
    atomic_fetch_add_explicit(&stats[0], 1u, memory_order_relaxed);
  }
}

// Coarse per-block difference without motion search, for the blend engine's
// scene-cut test.
kernel void block_difference(texture2d<float, access::sample> a [[texture(0)]],
                             texture2d<float, access::sample> b [[texture(1)]],
                             constant SearchParams& p [[buffer(0)]],
                             device atomic_uint* stats [[buffer(1)]],
                             uint2 gid [[thread_position_in_grid]]) {
  if (int(gid.x) >= p.field_size.x || int(gid.y) >= p.field_size.y) return;
  const float2 origin = float2(gid) * 8.0f;
  float s = 0.0f;
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      const float2 at = origin + float2(x, y) * 2.0f + 1.0f;
      s += fabs(a.sample(kLinear, at).r - b.sample(kLinear, at).r);
    }
  }
  if (s * (255.0f / 16.0f) > p.bad_sad) {
    atomic_fetch_add_explicit(&stats[0], 1u, memory_order_relaxed);
  }
}

// ------------------------------------------------------------ vector median

static float median9(thread float* v) {
  for (int i = 1; i < 9; ++i) {
    const float x = v[i];
    int j = i - 1;
    while (j >= 0 && v[j] > x) {
      v[j + 1] = v[j];
      --j;
    }
    v[j + 1] = x;
  }
  return v[4];
}

kernel void vector_median(texture2d<float, access::read> src [[texture(0)]],
                          texture2d<float, access::write> dst [[texture(1)]],
                          uint2 gid [[thread_position_in_grid]]) {
  const int2 size = int2(src.get_width(), src.get_height());
  if (int(gid.x) >= size.x || int(gid.y) >= size.y) return;
  float xs[9];
  float ys[9];
  int n = 0;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      const int2 q = clamp(int2(gid) + int2(dx, dy), int2(0), size - 1);
      const float4 v = src.read(uint2(q));
      xs[n] = v.x;
      ys[n] = v.y;
      ++n;
    }
  }
  const float4 center = src.read(gid);
  dst.write(float4(median9(xs), median9(ys), center.z, 0.0f), gid);
}

// ------------------------------------------------------------- interpolate

struct InterpParams {
  float t;
  int mode;             // 0 = motion compensated, 1 = blend
  float cut_fraction;   // unmatched-block share that marks a scene cut
  uint total_blocks;
  float occlusion_lo;   // prediction mismatch (0..1) where occlusion handling starts
  float occlusion_hi;   // ... and where it fully applies
  int bicubic;          // 1 => Catmull-Rom output sampling (sharper, slower)
};

// Catmull-Rom bicubic sampling from 9 bilinear taps. Bilinear alone visibly
// softens detail at half-pixel offsets, which is where interpolated frames
// sample most of the time.
static float4 sample_bicubic(texture2d<float, access::sample> tex, float2 p) {
  const float2 p1 = floor(p - 0.5f) + 0.5f;
  const float2 f = p - p1;
  const float2 w0 = f * (-0.5f + f * (1.0f - 0.5f * f));
  const float2 w1 = 1.0f + f * f * (-2.5f + 1.5f * f);
  const float2 w2 = f * (0.5f + f * (2.0f - 1.5f * f));
  const float2 w3 = f * f * (-0.5f + 0.5f * f);
  const float2 w12 = w1 + w2;
  const float2 p0 = p1 - 1.0f;
  const float2 p3 = p1 + 2.0f;
  const float2 p12 = p1 + w2 / w12;
  float4 r = tex.sample(kLinear, float2(p0.x, p0.y)) * w0.x * w0.y;
  r += tex.sample(kLinear, float2(p12.x, p0.y)) * w12.x * w0.y;
  r += tex.sample(kLinear, float2(p3.x, p0.y)) * w3.x * w0.y;
  r += tex.sample(kLinear, float2(p0.x, p12.y)) * w0.x * w12.y;
  r += tex.sample(kLinear, float2(p12.x, p12.y)) * w12.x * w12.y;
  r += tex.sample(kLinear, float2(p3.x, p12.y)) * w3.x * w12.y;
  r += tex.sample(kLinear, float2(p0.x, p3.y)) * w0.x * w3.y;
  r += tex.sample(kLinear, float2(p12.x, p3.y)) * w12.x * w3.y;
  r += tex.sample(kLinear, float2(p3.x, p3.y)) * w3.x * w3.y;
  return saturate(r);  // Catmull-Rom overshoots slightly at hard edges
}

static float4 sample_output(texture2d<float, access::sample> tex, float2 p,
                            constant InterpParams& params) {
  return params.bicubic != 0 ? sample_bicubic(tex, p) : tex.sample(kLinear, p);
}

// Bilinear vector lookup; field texels sit at 8x8 block centres.
static float2 field_at(texture2d<float, access::read> f, float2 p) {
  const float2 g = p / 8.0f - 0.5f;
  const float2 g0 = floor(g);
  const float2 fr = g - g0;
  const int2 size = int2(f.get_width(), f.get_height());
  const int2 i0 = clamp(int2(g0), int2(0), size - 1);
  const int2 i1 = clamp(int2(g0) + 1, int2(0), size - 1);
  const float2 v00 = f.read(uint2(i0.x, i0.y)).xy;
  const float2 v10 = f.read(uint2(i1.x, i0.y)).xy;
  const float2 v01 = f.read(uint2(i0.x, i1.y)).xy;
  const float2 v11 = f.read(uint2(i1.x, i1.y)).xy;
  return mix(mix(v00, v10, fr.x), mix(v01, v11, fr.x), fr.y);
}

struct Candidate {
  float2 pa;   // matching position in a (luma pixels)
  float2 pb;   // matching position in b
  float err;   // luma mismatch between the two, 0..1
};

static float patch_error(texture2d<float, access::sample> aY, texture2d<float, access::sample> bY,
                         float2 pa, float2 pb) {
  float e = 0.0f;
  const float2 offs[4] = {float2(-0.75f, -0.75f), float2(0.75f, -0.75f),
                          float2(-0.75f, 0.75f), float2(0.75f, 0.75f)};
  for (int i = 0; i < 4; ++i) {
    e += fabs(aY.sample(kLinear, pa + offs[i]).r - bY.sample(kLinear, pb + offs[i]).r);
  }
  return e * 0.25f;
}

// Forward field lives in a: a point x in a moves to x + v(x) in b. The output
// pixel p sits at x + t*v(x); solve for x with two fixed-point steps.
static Candidate forward_candidate(texture2d<float, access::read> fwd,
                                   texture2d<float, access::sample> aY,
                                   texture2d<float, access::sample> bY, float2 p, float t) {
  float2 v = field_at(fwd, p);
  float2 x = p - t * v;
  v = field_at(fwd, x);
  x = p - t * v;
  Candidate c;
  c.pa = x;
  c.pb = x + v;
  c.err = patch_error(aY, bY, c.pa, c.pb);
  return c;
}

// Backward field lives in b: a point y in b came from y + w(y) in a. The
// output pixel p sits at y + (1-t)*w(y).
static Candidate backward_candidate(texture2d<float, access::read> bwd,
                                    texture2d<float, access::sample> aY,
                                    texture2d<float, access::sample> bY, float2 p, float t) {
  float2 w = field_at(bwd, p);
  float2 y = p - (1.0f - t) * w;
  w = field_at(bwd, y);
  y = p - (1.0f - t) * w;
  Candidate c;
  c.pb = y;
  c.pa = y + w;
  c.err = patch_error(aY, bY, c.pa, c.pb);
  return c;
}

// Combines the two predictions. fa/fb: samples at the forward candidate's
// points in a and b; ba/bb: same for the backward candidate.
static float4 combine(Candidate f, Candidate b, float t, constant InterpParams& p,
                      float4 fa, float4 fb, float4 ba, float4 bb) {
  const float wf = 1.0f / (f.err * f.err + 1e-4f);
  const float wb = 1.0f / (b.err * b.err + 1e-4f);
  const float4 motion = (wf * mix(fa, fb, t) + wb * mix(ba, bb, t)) / (wf + wb);

  // Neither prediction agrees: the content is visible in only one frame.
  // Take it from the nearer frame, using the better-matching vector.
  const bool forward_better = f.err <= b.err;
  const float4 near_a = forward_better ? fa : ba;
  const float4 near_b = forward_better ? fb : bb;
  const float4 one_sided = t < 0.5f ? near_a : near_b;
  const float k = smoothstep(p.occlusion_lo, p.occlusion_hi, min(f.err, b.err));
  return mix(motion, one_sided, k);
}

kernel void interpolate_frame(texture2d<float, access::sample> aY [[texture(0)]],
                              texture2d<float, access::sample> aUV [[texture(1)]],
                              texture2d<float, access::sample> bY [[texture(2)]],
                              texture2d<float, access::sample> bUV [[texture(3)]],
                              texture2d<float, access::read> fwd [[texture(4)]],
                              texture2d<float, access::read> bwd [[texture(5)]],
                              texture2d<float, access::write> outY [[texture(6)]],
                              texture2d<float, access::write> outUV [[texture(7)]],
                              constant InterpParams& p [[buffer(0)]],
                              device const uint* stats [[buffer(1)]],
                              uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= outUV.get_width() || gid.y >= outUV.get_height()) return;
  const uint2 luma_size = uint2(outY.get_width(), outY.get_height());

  float t = p.t;
  const bool cut = p.total_blocks > 0 &&
                   float(stats[0]) > p.cut_fraction * float(p.total_blocks);
  if (cut) t = t < 0.5f ? 0.0f : 1.0f;

  if (p.mode == 1 || cut) {
    for (uint j = 0; j < 2; ++j) {
      for (uint i = 0; i < 2; ++i) {
        const uint2 lp = gid * 2 + uint2(i, j);
        if (lp.x >= luma_size.x || lp.y >= luma_size.y) continue;
        const float2 at = float2(lp) + 0.5f;
        outY.write(float4(mix(aY.sample(kLinear, at).r, bY.sample(kLinear, at).r, t)), lp);
      }
    }
    const float2 at = float2(gid) + 0.5f;
    outUV.write(mix(aUV.sample(kLinear, at), bUV.sample(kLinear, at), t), gid);
    return;
  }

  for (uint j = 0; j < 2; ++j) {
    for (uint i = 0; i < 2; ++i) {
      const uint2 lp = gid * 2 + uint2(i, j);
      if (lp.x >= luma_size.x || lp.y >= luma_size.y) continue;
      const float2 at = float2(lp) + 0.5f;
      const Candidate f = forward_candidate(fwd, aY, bY, at, t);
      const Candidate b = backward_candidate(bwd, aY, bY, at, t);
      const float4 y = combine(f, b, t, p,
                               sample_output(aY, f.pa, p), sample_output(bY, f.pb, p),
                               sample_output(aY, b.pa, p), sample_output(bY, b.pb, p));
      outY.write(float4(y.r), lp);
    }
  }

  // Chroma: vectors from the centre of this 2x2 luma block; chroma texel
  // coordinates are luma coordinates halved.
  const float2 centre = float2(gid) * 2.0f + 1.0f;
  const Candidate f = forward_candidate(fwd, aY, bY, centre, t);
  const Candidate b = backward_candidate(bwd, aY, bY, centre, t);
  const float4 uv = combine(f, b, t, p,
                            sample_output(aUV, f.pa * 0.5f, p), sample_output(bUV, f.pb * 0.5f, p),
                            sample_output(aUV, b.pa * 0.5f, p), sample_output(bUV, b.pb * 0.5f, p));
  outUV.write(uv, gid);
}
