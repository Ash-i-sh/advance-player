# Advance Player — Product & Technical Plan

Status: **M0 in progress** — engine foundation built and measured on the M4
(see README.md). Last updated 2026-10-29.

## 1. Product summary

A commercial video player that makes 24 fps films look smooth (60/120 fps) and
improves picture and sound, adapting to whatever hardware it runs on.

- **Platforms:** macOS first (develop and test on the M4 Mac), then Windows once
  the Mac product is finished. Android and iOS later (v2). No TV/Linux for now.
- **Modes:** live smoothing while watching **and** offline convert/export.
- **Sources:** local files (primary), network (links, HLS/DASH, NAS/SMB), and
  web video such as YouTube, played inside our player without ads (see §6).
- **Not supported, ever:** DRM services (Netflix, Prime, Hotstar, Disney+) — see §6a.

## 2. What we learned from existing apps

| App | Approach | Lesson for us |
|---|---|---|
| SVP | Plug-in to mpv/VLC via VapourSynth; classic motion vectors live, RIFE on strong GPUs; paid FRC on Android | Live = fast classic method on most hardware; plug-in model forces slow copy-back decoding → we own the pipeline |
| Topaz Video AI | AI models (Chronos/Apollo/Aion), offline only, ~7–10 fps on M1 Ultra | Best AI quality is an export feature |
| Flowframes | RIFE via CUDA or ncnn/Vulkan, offline, Windows | One model can run on every GPU vendor |
| Lossless Scaling | Captures any window, generates frames, sold on Steam | "Boost any window" works for browser video; refresh-rate matching is critical |
| NVIDIA FRUC | Hardware optical flow + CUDA, Win/Linux, Turing+ | Free fast engine on NVIDIA PCs |
| Apple VideoToolbox | ML frame-rate conversion, macOS 15.4+ / iOS 26 | Free fast engine on Apple |
| Phone MEMC | Display chip, not accessible to apps | Mobile must use GPU/NPU ourselves (v2) |
| mpv smoothmotion | Frame blending + display-resample | Universal fallback |

## 3. Key decisions

| Decision | Choice | Why |
|---|---|---|
| UI | **Flutter** (one UI for Win/Mac now, Android/iOS later) | One codebase for 4 platforms, BSD licence, fine for a media app UI |
| Engine | **Own C++20 core**, not mpv/VLC | Direct frame access for interpolation, no copy-back penalty, avoids GPL |
| Decode | FFmpeg (**LGPL build**) + hardware decode (D3D11VA / VideoToolbox) | Plays everything; LGPL allows selling |
| Video output | Core owns a native surface (D3D11/12 swapchain, CAMetalLayer) under the Flutter UI | Precise frame pacing; Flutter textures add latency (verify in spike) |
| Colour / HDR | libplacebo (LGPL) | Proven tone mapping, debanding, LUTs |
| Shaders | Authored once (Slang), compiled to HLSL/SPIR-V/Metal | Single source for all GPUs |
| AI runtime | ONNX Runtime: DirectML (any Windows GPU/NPU), TensorRT (NVIDIA), CoreML (Apple) | One model format, per-device acceleration |
| Native engines | Apple VideoToolbox frame-rate conversion, NVIDIA FRUC | Fast, free, hardware-accelerated |
| Encode (export) | Hardware only: VideoToolbox, NVENC, AMF, QSV, Media Foundation | Avoids GPL x264/x265 and most patent exposure |

## 4. Hardware-adaptive design (no hard-coded device rules)

1. **Probe** at first launch and after OS/driver change: OS, GPU API and vendor,
   NPU, RAM/VRAM, hardware decoders, display refresh rate and HDR, power/thermal state.
2. **Benchmark** every engine the device supports on a built-in clip (ms/frame at
   720p/1080p/4K). Results cached per device.
3. **Select** the best tier that fits the frame budget (1000 / target fps ms):

   | Tier | Engine | Typical devices |
   |---|---|---|
   | L5 | Large AI model — export only | Any |
   | L4 | Small AI model (GPU/NPU) | Strong PCs, Apple Silicon |
   | L3 | Apple VideoToolbox / NVIDIA FRUC | macOS 15.4+, RTX PCs |
   | L2 | Classic motion-vector MEMC (our GPU shader) | Most GPUs |
   | L1 | Frame blending + display sync | Everything |
   | L0 | Off | Everything |

4. **Govern at runtime:** on dropped frames step down (flow resolution → smaller
   model → lower target fps → lower tier); step up with hysteresis when there is
   headroom; react to battery saver and thermal state.
5. **Engines are plug-ins** behind one interface (`supported()`, `benchmark()`,
   `interpolate(f0, f1, t)`), so new engines need no flow changes.

The same probe/benchmark/govern loop drives upscaling, denoise and tone mapping.

Smoothing quality rules: target = display refresh (60 Hz → 60, 120 Hz → 120);
no interpolation across scene cuts; protect on-screen text/credits; user
strength slider to avoid the "soap-opera effect".

## 5. Version 1 scope

**v1.0 (macOS first, then Windows)**
- Local playback: all common formats, embedded/external subtitles, audio tracks,
  chapters, resume, frame-step, screenshots.
- Adaptive smoothing L0–L4, refresh-rate matching, scene-cut detection, strength slider.
- Picture: presets, brightness/contrast/saturation/temperature, LUT (.cube) import,
  HDR→SDR tone mapping, deband, sharpen; **split-screen before/after**.
- Export queue with L5 quality and hardware encoding.
- Network: HTTP(S) links, HLS/DASH, SMB/NAS.
- "Open link" for YouTube and other sites, ad-free, through our pipeline (§6).
- Stats overlay, hardware auto-tune, licence activation.

**v1.1** AI upscaling; dialogue boost, night mode, loudness levelling; auto
black-bar crop; skip intro/credits.

**v1.2** AI subtitles (Whisper) and translation; library with posters;
Jellyfin/Plex sources.

**v2** Android and iOS (NPU-friendly model, mobile governor).

Monetisation suggestion: free tier (L0–L2, basic colour) + Pro (AI smoothing,
export, AI upscaling) — same model as SVP and Topaz.

## 6. YouTube and web video

**Chosen: option B — open links in our player.** Paste or drop a link; the app
resolves the stream with yt-dlp (supports YouTube and ~1,800 other sites) and
plays it through our own pipeline, so smoothing, colour and audio all apply.
Ads are separate requests and are never fetched.

Design rules for this feature:
- **Generic "Open link" source**, not a YouTube-branded feature. No YouTube name
  or logo in the app or marketing (Google's 2022 cease-and-desist to Vanced
  demanded exactly that).
- **yt-dlp lives outside the app binary** and is updated on its own channel.
  YouTube changes its protocol often (PO tokens, SABR streaming in 2026), so
  extraction can break any week; users must get fixes without an app release.
- **Graceful failure:** if extraction fails, offer "open in browser" instead of a broken player.
- **Distribution:** sell direct (own site), not the Mac App Store — Apple is likely
  to reject it.
- **Legal review before launch is required.** Ad-free playback conflicts with
  YouTube's Terms of Service and is the most legally exposed part of the product.

Option A (boost any window via ScreenCaptureKit / Windows Graphics Capture) is
kept as a possible later addition for other apps' video.

## 6a. Why Netflix and other DRM services cannot be supported

- These services encrypt video and only give decryption keys to licensed players
  (FairPlay in Safari/Apple TV app, Widevine in Chrome/Edge, PlayReady on Windows).
- Decrypted frames travel a **protected path**: the OS and GPU mark the video
  surface as protected, so capture APIs (ScreenCaptureKit, Windows Graphics
  Capture, OBS) receive black frames, and HDCP encrypts the cable to the display.
- Getting the frames anyway means circumventing DRM. That is illegal under
  anti-circumvention laws in most countries, breaks the services' terms, and
  would get the product taken down. **Out of scope.**
- What does work legitimately: motion smoothing built into the TV or monitor,
  which runs after decryption inside the display. The app should detect protected
  (black) content and explain this to the user.

## 7. Selling the product — constraints

- Only LGPL/MIT/BSD/Apache components, dynamically linked. No mpv GPL build,
  no MVTools, no x264/x265.
- Check every AI model's licence **and** its training-data licence before shipping.
- Dolby Vision / Atmos / TrueHD / DTS and HEVC/AAC: rely on OS decoders or
  passthrough; get licensing advice.
- Code signing: Apple Developer Program (notarisation) and a Windows code-signing certificate.
- Sales channel to decide: own website with a merchant of record (handles tax),
  Steam, and/or Microsoft Store / Mac App Store.

## 8. Milestones

**Phase 1 — macOS (develop and test on the M4 Mac)**

| # | Milestone | Exit criteria |
|---|---|---|
| M0 | Setup + 3 spikes: frame pacing under Flutter UI, engine benchmarks on the M4 (VideoToolbox, Core ML RIFE, classic MEMC), yt-dlp link playback | Go/no-go on architecture |
| M1 | Player core: decode, audio, A/V sync, subtitles, Flutter UI | Smooth 4K playback on the M4 |
| M2 | Smoothing engines L1–L4 (Apple backends) + probe/benchmark/governor | No dropped frames at chosen tier |
| M3 | Colour pipeline + before/after | |
| M4 | Export queue (L5) | |
| M5 | Network sources + "Open link" | |
| M6 | Licensing, signing/notarisation, Mac beta | Shippable Mac v1.0 |

**Phase 2 — Windows (after the Mac product is finished)**

| # | Milestone | Exit criteria |
|---|---|---|
| W1 | Windows backends: D3D presenter, D3D11VA decode, DirectML/TensorRT, NVIDIA FRUC, hardware encoders | Same tiers working on the PC |
| W2 | Installer, code signing, Windows beta | Shippable Windows v1.0 |

**M0 progress (2026-10-29, Apple M4, macOS 15.2)**

- Done: LGPL FFmpeg build; device probe; L1/L2 Metal engines; benchmark-based
  selection; governor; `advp convert` (hardware decode → GPU → hardware encode,
  8/10-bit, audio kept); unit tests; quality benchmark.
- Measured at 1080p: `motion` 5.5 ms per source pair + 1.9 ms per new frame →
  44% of budget at 24→100 fps, so real-time is comfortable. At 4K the selector
  falls back to `motion-fast` automatically.
- Quality (PSNR of rebuilt frames, `tests/quality.sh`): see "Motion tuning" below.
- Conversion speed: 1080p 23.976 → 59.94 at 2.6–3.3× real time.
- Frame-pacing spike (native AppKit player, `advp play`), M4 with 100 Hz and
  75 Hz displays, 23.976 fps sources, 6–8 s runs:

  | Source | Engine | Smoothing per refresh (mean / p99) | Missed refreshes |
  |---|---|---|---|
  | 1080p | off | — | 4 of 588 |
  | 1080p | motion (auto pick) | 2.9 / 5.0 ms | 2 of 793 |
  | 4K 10-bit | blend (auto pick) | 2.5 / 3.1 ms | 6 of 788 |
  | 4K 10-bit | motion-fast (forced) | 6.7 / 12.6 ms | 134 of 588 |
  | 4K 10-bit | motion (forced) | 15.3 / 25.5 ms | 274 of 590 |

  The remaining misses come from the window opening (they also happen with
  smoothing off). The benchmark's predictions held: at 4K/100 Hz it ruled
  out `motion-fast` (592 of 500 ms/s budget), and playing it misses frames.
  Two things mattered: reading the refresh rate from the window's own screen
  (multi-monitor), and analysing the next source pair in the GPU's idle time
  instead of on the refresh that needs it (p99 9.9 → 3.2 ms for motion-fast at 1080p).
- Spike result: go for a native presenter (CAMetalLayer + display link) with
  smoothing at the display rate. Still open: the same pacing inside Flutter
  (needs Xcode), audio clock and A/V sync (M1), and cheaper 4K motion, e.g.
  skipping the full-resolution search level.
- Waiting on owner: macOS 15.4+ and Xcode (Flutter UI, VideoToolbox L3 engine).
  Free disk space (~10 GB) must be raised first.
- Fixed output rate (`advp play --fps fit:60 | 60 | display`): each frame is
  made once, ahead of time, while the previous one shows. On the 100 Hz screen
  `fit:60` runs 50 fps with every frame held exactly two refreshes (0 missed,
  0.000 ms timing error at 1080p), and 4K 10-bit keeps `motion-fast` where
  the full refresh rate could only afford `blend`. The governor now judges
  load per source frame and ignores the first 0.5 s after a start or seek.
- Motion tuning (2026-10-29). The benchmark was fixed first: its frame
  pairing was off by one frame on some timestamps (the "26–29 dB outliers"
  were that), and ffmpeg's whole-pixel crop made the reference pan itself
  inaccurate; the pan is now rendered exactly (`tests/make_pan.py`).
  Engine changes:
  per-pixel vector selection among the smooth field, zero motion and the
  four nearest raw block vectors; matches scored on half-resolution luma
  (robust to grain); a forward/backward consistency check picks the right
  frame for covered/uncovered areas; zero motion is the search fallback with a
  5% margin.

  | Clip | Engine | Before | After |
  |---|---|---|---|
  | Pan (exact sub-pixel) | motion | 41.02 dB | 40.89 dB |
  | Pan | motion-fast | 38.94 dB | 38.86 dB |
  | Test card (objects, occlusion, noise) | motion | 34.28 dB (worst 32.78) | 34.72 dB (worst 33.03) |
  | Test card | motion-fast | 34.26 dB | 34.70 dB |

  Ceilings measured with the true motion: on the pan, luma is within ~1 dB
  of perfect vectors with our Catmull-Rom output and chroma is at the 4:2:0
  limit, so pans have little left without a sharper filter (Lanczos: +1.8 dB,
  ~4x the taps; a candidate for the export engine). Tried and rejected:
  quarter-pel (<0.1 dB), half-pel and bicubic output for motion-fast (+0.1 dB
  for +14% cost; +1.2 dB for 2x per-frame cost). The test card's noise
  squares change every frame, so no method can predict them; its remaining
  error is mostly that. Speed not re-measured after the last change (the
  machine was under screen-sharing load).
- Real footage (2026-10-30): a 2.5-minute 4K film trailer (VFX, fire,
  crowds, fades, ~70 cuts), scored with `tests/film_quality.sh` (24 → 12 →
  24 fps at 1080p, 1645 rebuilt frames). On ordinary shots motion was already
  sound (+3.5 dB over blend); the failures were all in three situations:
  - Fades through black: every block looked unmatched, the search chased
    chance matches and the occlusion fallback copied warped content. Matching
    now also scores the reference at the other frame's average brightness
    (gain about black, from a per-frame mean) and keeps the better score;
    deep in a fade (one frame nearly black, or >3x brightness change) the
    pair is blended. The extra score runs only when brightness changed.
  - Glow or light sweeping over a still logo: the unscaled score keeps those
    matching (the gain alone made them worse).
  - Missed scene cuts: chance matches in busy texture (crowds) kept most
    blocks "matched", so the 55% unmatched threshold missed most cuts and the
    engine morphed two shots together. Measured on 3439 real 24 fps pairs with
    68 hand-checked cuts, 12% misses 2 cuts and flags 27 other pairs (fast
    motion, flashes), where a frame repeats. Tried and rejected: unmatched
    relative to block texture (grain makes ordinary pairs look unmatched) and
    colour-histogram change (flashes change it more than some cuts do).

  | Trailer, motion engine | Before | After |
  |---|---|---|
  | Mean PSNR of rebuilt frames | 37.72 dB | 38.15 dB |
  | Frames >6 dB worse than blend | 24 | 3 |
  | Frames >10 dB worse than blend | 11 | 1 |

  Synthetic clips unchanged (pan 40.89, card 34.72 dB). Still weak: a
  coloured light sweeping across a still logo (frame 13 of the trailer, 16 dB
  under blend), and cuts from a dark shot to a bright one. Speed: no slower than
  before in an A/B run, but absolute timings again under screen-sharing load.
- Audio in the player (M1 start, 2026-10-30): one demux pass feeds both
  decoders; audio is decoded to float stereo (surround downmixed) and played
  through Core Audio's default output, which follows the system's device.
  The video clock stays the master, because it is tied to display refreshes
  and keeps frame pacing exact; audio follows it, as mpv's smoothest mode
  does. Each output buffer is placed by its host time plus the device's
  latency (latency + I/O buffer + safety offset, as mpv counts it, plus the
  unit's own rate conversion). Clock drift is removed one sample per buffer;
  errors over 5 ms (start, seek, resume, a skipped buffer) are fixed at once.
  While audio runs low the decoder may read up to 24 video frames ahead, for
  loosely interleaved files.
  Measured against the video clock (Opus, AAC, 5.1 AC-3, 44.1 kHz FLAC):
  mean 0.01–0.05 ms, max under 0.1 ms after start-up, no underruns, through
  scripted pause, seek, mute and volume changes. Not measurable here: the
  real acoustic offset (needs ears or a microphone; a flash-and-beep clip is
  in the README) and display latency, which is not allowed for. The audio
  thread only try-locks the buffer; a lock-free ring would remove the rare
  skipped buffer.
- Waiting on owner: macOS 15.4+ and Xcode (Flutter UI, VideoToolbox L3 engine).
  Free disk space (~10 GB) must be raised first.
- Next without Xcode: yt-dlp link spike, subtitles and audio-track choice,
  temporal vector prediction (steadier vectors between pairs), cheaper 4K
  motion.

Keeping Windows cheap later: platform-specific code sits behind interfaces from
day one, and a Windows CI job compiles the shared core on every change so it
never drifts Mac-only. Windows-only backends are written in Phase 2, when they
can be tested on real hardware, not blind.

## 9. Prerequisites (owner's steps)

- Mac: update to macOS 15.4 or later, install Xcode and Flutter; fix Homebrew
  ffmpeg (currently fails to launch).
- PC: not needed until Phase 2; share specs then.

## References

- SVP manual: https://svp-team.com/wiki/Manual:SVP
- SVP RIFE: https://svp-team.com/wiki/RIFE_AI_interpolation
- Topaz Video: https://www.topazlabs.com/topaz-video
- Flowframes: https://github.com/n00mkrad/flowframes
- NVIDIA FRUC: https://docs.nvidia.com/video-technologies/optical-flow-sdk/nvfruc-programming-guide/index.html
- Apple frame-rate conversion: https://developer.apple.com/documentation/videotoolbox/vtframerateconversionconfiguration
- ONNX Runtime execution providers: https://onnxruntime.ai/docs/execution-providers/
- ANVIL (mobile NPU interpolation): https://arxiv.org/html/2603.26835
- Capture APIs and DRM: https://sageinfinity.github.io/docs/FAQ/dxgiwgc
- yt-dlp PO token guide: https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide
- Vanced shutdown: https://9to5google.com/2022/03/13/youtube-vanced-shutting-down/
