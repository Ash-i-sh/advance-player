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
  int subpel;              // sub-pixel refinement: 0 none, 1 half-pel, 2 quarter-pel
  float bad_sad;           // mean abs difference (0..255) meaning "no match"
  int count_stats;         // 1 => count unmatched blocks into stats[0]
  float new_penalty;       // relative SAD surcharge on any non-zero vector
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

// Zero motion is the fallback: it pays nothing, and any other vector must
// beat it by the relative surcharge plus its distance from the prediction.
// Where nothing really matches (grain, rain, changing noise), a chance alias
// or a vector inherited from neighbouring motion would drag content around;
// staying still is the far milder mistake.
static void try_vector(thread const float* blk, texture2d<float, access::sample> ref,
                       float2 origin, float2 v, float2 pred, float lambda, float surcharge,
                       thread float2& best, thread float& best_cost, thread float& best_sad) {
  const float sad = block_sad(blk, ref, origin, v);
  const float2 d = abs(v - pred);
  const bool zero = all(v == 0.0f);
  const float cost = zero ? sad : sad * (1.0f + surcharge) + lambda * (d.x + d.y);
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
        try_vector(blk, ref, origin, float2(dx, dy), pred, p.lambda, p.new_penalty, best,
                   best_cost, best_sad);
      }
    }
  } else {
    const int2 c = int2(gid) / 2;
    const int2 cs = p.coarse_field_size;
    pred = coarse_vector(coarse, c, cs);
    try_vector(blk, ref, origin, pred, pred, p.lambda, p.new_penalty, best, best_cost, best_sad);
    try_vector(blk, ref, origin, float2(0.0f), pred, p.lambda, p.new_penalty, best, best_cost, best_sad);
    try_vector(blk, ref, origin, coarse_vector(coarse, c + int2(1, 0), cs), pred, p.lambda,
               p.new_penalty, best, best_cost, best_sad);
    try_vector(blk, ref, origin, coarse_vector(coarse, c + int2(-1, 0), cs), pred, p.lambda,
               p.new_penalty, best, best_cost, best_sad);
    try_vector(blk, ref, origin, coarse_vector(coarse, c + int2(0, 1), cs), pred, p.lambda,
               p.new_penalty, best, best_cost, best_sad);
    try_vector(blk, ref, origin, coarse_vector(coarse, c + int2(0, -1), cs), pred, p.lambda,
               p.new_penalty, best, best_cost, best_sad);
  }

  for (int i = 0; i < p.refine_iters; ++i) {
    const float2 center = best;
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0) continue;
        try_vector(blk, ref, origin, center + float2(dx, dy), pred, p.lambda,
                   p.new_penalty, best, best_cost, best_sad);
      }
    }
  }
  // Sub-pixel refinement: half-pel, then quarter-pel when subpel is 2.
  for (int level = 1; level <= p.subpel; ++level) {
    const float step = level == 1 ? 0.5f : 0.25f;
    const float2 center = best;
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0) continue;
        try_vector(blk, ref, origin, center + step * float2(dx, dy), pred, p.lambda,
                   p.new_penalty, best, best_cost, best_sad);
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
  float candidate_penalty;  // extra error charged to the raw-vector candidates
  float score_scale;        // scoring texture size relative to full resolution
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
  int kind;    // 0 forward field, 1 backward field, 2 constant vector
};

// Mismatch between a at pa and b at pb (full-resolution luma coordinates),
// measured on the half-resolution pyramid level (scale 0.5) when there is
// one: 2x2 averaging takes out most grain and noise, which changes every
// frame and would otherwise make flat areas look like better matches.
static float patch_error(texture2d<float, access::sample> aL, texture2d<float, access::sample> bL,
                         float2 pa, float2 pb, float scale) {
  float e = 0.0f;
  const float2 offs[4] = {float2(-0.75f, -0.75f), float2(0.75f, -0.75f),
                          float2(-0.75f, 0.75f), float2(0.75f, 0.75f)};
  for (int i = 0; i < 4; ++i) {
    e += fabs(aL.sample(kLinear, pa * scale + offs[i]).r - bL.sample(kLinear, pb * scale + offs[i]).r);
  }
  return e * 0.25f;
}

// Forward field lives in a: a point x in a moves to x + v(x) in b. The output
// pixel p sits at x + t*v(x); solve for x with two fixed-point steps.
static Candidate forward_points(texture2d<float, access::read> fwd, float2 p, float t) {
  float2 v = field_at(fwd, p);
  float2 x = p - t * v;
  v = field_at(fwd, x);
  x = p - t * v;
  Candidate c;
  c.pa = x;
  c.pb = x + v;
  c.err = 0.0f;
  c.kind = 0;
  return c;
}

static Candidate forward_candidate(texture2d<float, access::read> fwd,
                                   texture2d<float, access::sample> aL,
                                   texture2d<float, access::sample> bL, float2 p, float t,
                                   float scale) {
  Candidate c = forward_points(fwd, p, t);
  c.err = patch_error(aL, bL, c.pa, c.pb, scale);
  return c;
}

// Backward field lives in b: a point y in b came from y + w(y) in a. The
// output pixel p sits at y + (1-t)*w(y).
static Candidate backward_points(texture2d<float, access::read> bwd, float2 p, float t) {
  float2 w = field_at(bwd, p);
  float2 y = p - (1.0f - t) * w;
  w = field_at(bwd, y);
  y = p - (1.0f - t) * w;
  Candidate c;
  c.pb = y;
  c.pa = y + w;
  c.err = 0.0f;
  c.kind = 1;
  return c;
}

static Candidate backward_candidate(texture2d<float, access::read> bwd,
                                    texture2d<float, access::sample> aL,
                                    texture2d<float, access::sample> bL, float2 p, float t,
                                    float scale) {
  Candidate c = backward_points(bwd, p, t);
  c.err = patch_error(aL, bL, c.pa, c.pb, scale);
  return c;
}

// The candidate's matching points for pixel at. Field candidates are solved
// again per pixel (the field is smooth, so this keeps pans sharp); constant
// vectors shift like the block they were chosen for.
static void points_for(Candidate c, float2 at, float2 q, float t,
                       texture2d<float, access::read> fwd, texture2d<float, access::read> bwd,
                       thread float2& pa, thread float2& pb) {
  if (c.kind == 0) {
    const Candidate s = forward_points(fwd, at, t);
    pa = s.pa;
    pb = s.pb;
  } else if (c.kind == 1) {
    const Candidate s = backward_points(bwd, at, t);
    pa = s.pa;
    pb = s.pb;
  } else {
    pa = at + (c.pa - q);
    pb = at + (c.pb - q);
  }
}

// Constant motion m (a -> b) through point p: a at p - t*m, b at p + (1-t)*m.
// penalty: added error, so these extra candidates win only when clearly better.
static Candidate motion_candidate(texture2d<float, access::sample> aL,
                                  texture2d<float, access::sample> bL, float2 p, float2 m,
                                  float t, float penalty, float scale) {
  Candidate c;
  c.pa = p - t * m;
  c.pb = p + (1.0f - t) * m;
  c.err = patch_error(aL, bL, c.pa, c.pb, scale) + penalty;
  c.kind = 2;
  return c;
}

// Keeps the two lowest-error candidates, best first.
static void consider(Candidate c, thread Candidate& best1, thread Candidate& best2) {
  if (c.err < best1.err) {
    best2 = best1;
    best1 = c;
  } else if (c.err < best2.err) {
    best2 = c;
  }
}

// Where content is visible in only one frame (no candidate matches both),
// which frame to take it from: +1 = a (it is being covered by time b),
// -1 = b (it is being uncovered), 0 = unknown. A point whose vector the
// other field does not send back is the visible-on-one-side case.
static float occlusion_side(texture2d<float, access::read> fwd, texture2d<float, access::read> bwd,
                            Candidate f, Candidate b) {
  const float covered = length((f.pb - f.pa) + field_at(bwd, f.pb));
  const float uncovered = length((b.pa - b.pb) + field_at(fwd, b.pa));
  return smoothstep(1.5f, 4.0f, covered) - smoothstep(1.5f, 4.0f, uncovered);
}

// Blends the two best candidates by how well each matches. c1a/c1b: samples
// at the first candidate's points in a and b; c2a/c2b: the second's.
static float4 blend_candidates(Candidate c1, Candidate c2, float t, float4 c1a, float4 c1b,
                               float4 c2a, float4 c2b) {
  const float w1 = 1.0f / (c1.err * c1.err + 1e-4f);
  const float w2 = 1.0f / (c2.err * c2.err + 1e-4f);
  return (w1 * mix(c1a, c1b, t) + w2 * mix(c2a, c2b, t)) / (w1 + w2);
}

// The one-frame value for an occluded pixel. only_a/only_b: samples taken
// along the forward / backward field; near: the best candidate's sample in
// the nearer frame, used when the side is unclear.
static float4 one_sided(float side, float4 only_a, float4 only_b, float4 near) {
  const float4 chosen = side > 0.0f ? only_a : only_b;
  return mix(near, chosen, saturate(fabs(side)));
}

kernel void interpolate_frame(texture2d<float, access::sample> aY [[texture(0)]],
                              texture2d<float, access::sample> aUV [[texture(1)]],
                              texture2d<float, access::sample> bY [[texture(2)]],
                              texture2d<float, access::sample> bUV [[texture(3)]],
                              texture2d<float, access::read> fwd [[texture(4)]],
                              texture2d<float, access::read> bwd [[texture(5)]],
                              texture2d<float, access::write> outY [[texture(6)]],
                              texture2d<float, access::write> outUV [[texture(7)]],
                              texture2d<float, access::sample> aL [[texture(8)]],
                              texture2d<float, access::sample> bL [[texture(9)]],
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

  // Vector selection for this 2x2 luma block (one chroma texel). Bilinear
  // field vectors blur motion across object boundaries, so the block's own
  // and its neighbours' raw vectors, and zero motion, compete as well; the
  // two that best explain both frames are kept.
  const float2 q = float2(gid) * 2.0f + 1.0f;
  const Candidate fc = forward_candidate(fwd, aL, bL, q, t, p.score_scale);
  const Candidate bc = backward_candidate(bwd, aL, bL, q, t, p.score_scale);
  Candidate best1 = fc;
  Candidate best2 = bc;
  if (best2.err < best1.err) {
    best1 = bc;
    best2 = fc;
  }
  consider(motion_candidate(aL, bL, q, float2(0.0f), t, p.candidate_penalty, p.score_scale), best1, best2);
  const float2 g = q / 8.0f - 0.5f;
  const int2 size = int2(fwd.get_width(), fwd.get_height());
  const int2 g0 = int2(floor(g));
  for (int dy = 0; dy <= 1; ++dy) {
    for (int dx = 0; dx <= 1; ++dx) {
      const uint2 at = uint2(clamp(g0 + int2(dx, dy), int2(0), size - 1));
      consider(motion_candidate(aL, bL, q, fwd.read(at).xy, t, p.candidate_penalty, p.score_scale), best1, best2);
      consider(motion_candidate(aL, bL, q, -bwd.read(at).xy, t, p.candidate_penalty, p.score_scale), best1, best2);
    }
  }
  const float2 da1 = best1.pa - q, db1 = best1.pb - q;
  const float2 da2 = best2.pa - q, db2 = best2.pb - q;

  // Occlusion: how much of this block is visible in only one frame, and which.
  const float k = smoothstep(p.occlusion_lo, p.occlusion_hi, best1.err);
  const float side = k > 0.0f ? occlusion_side(fwd, bwd, fc, bc) : 0.0f;
  const float2 da_only = fc.pa - q;
  const float2 db_only = bc.pb - q;

  for (uint j = 0; j < 2; ++j) {
    for (uint i = 0; i < 2; ++i) {
      const uint2 lp = gid * 2 + uint2(i, j);
      if (lp.x >= luma_size.x || lp.y >= luma_size.y) continue;
      const float2 at = float2(lp) + 0.5f;
      float2 pa1, pb1, pa2, pb2;
      points_for(best1, at, q, t, fwd, bwd, pa1, pb1);
      points_for(best2, at, q, t, fwd, bwd, pa2, pb2);
      const float4 a1 = sample_output(aY, pa1, p);
      const float4 b1 = sample_output(bY, pb1, p);
      float4 y = blend_candidates(best1, best2, t, a1, b1, sample_output(aY, pa2, p),
                                  sample_output(bY, pb2, p));
      if (k > 0.0f) {
        const float4 only = one_sided(side, sample_output(aY, at + da_only, p),
                                      sample_output(bY, at + db_only, p), t < 0.5f ? a1 : b1);
        y = mix(y, only, k);
      }
      outY.write(float4(y.r), lp);
    }
  }

  // Chroma texel coordinates are luma coordinates halved.
  const float2 c = float2(gid) + 0.5f;
  const float4 ua1 = sample_output(aUV, c + da1 * 0.5f, p);
  const float4 ub1 = sample_output(bUV, c + db1 * 0.5f, p);
  float4 uv = blend_candidates(best1, best2, t, ua1, ub1, sample_output(aUV, c + da2 * 0.5f, p),
                               sample_output(bUV, c + db2 * 0.5f, p));
  if (k > 0.0f) {
    const float4 only = one_sided(side, sample_output(aUV, c + da_only * 0.5f, p),
                                  sample_output(bUV, c + db_only * 0.5f, p), t < 0.5f ? ua1 : ub1);
    uv = mix(uv, only, k);
  }
  outUV.write(uv, gid);
}
