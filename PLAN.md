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
- Quality (PSNR of rebuilt frames): camera pan 37.9 dB vs 25.9 dB for plain
  frame repeat (+12 dB); moving objects 34.2 vs 26.9 dB (+7.3 dB).
- Conversion speed: 1080p 23.976 → 59.94 at 2.6–3.3× real time.
- Waiting on owner: macOS 15.4+ and Xcode (frame-pacing spike, Flutter UI,
  VideoToolbox L3 engine). Free disk space (~10 GB) must be raised first.
- Next without Xcode: native AppKit test player for the frame-pacing spike,
  yt-dlp link spike, motion-search tuning (temporal candidates, vector projection).

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
