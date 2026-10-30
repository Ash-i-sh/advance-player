# Advance Player

A video player that smooths 24 fps films to the display's refresh rate and
improves picture and sound, adapting to the hardware it runs on.
Product and technical plan: [PLAN.md](PLAN.md).

## Status

Milestone M0 (engine foundation), macOS:

- LGPL FFmpeg build with VideoToolbox hardware decode/encode.
- Device probe: GPU, memory, hardware decoders, displays and refresh rates,
  power and thermal state.
- Smoothing engines on Metal: `blend` (L1), `motion-fast` and `motion` (L2,
  hierarchical block matching with bidirectional motion compensation,
  occlusion handling and scene-cut detection).
- Benchmark-driven engine selection and a runtime governor (no per-device rules).
- `advp convert`: file → smoothed file, zero-copy hardware decode → GPU →
  hardware encode, audio and subtitles kept, 8-bit and 10-bit.
- `advp play`: test player window. Smooths live to the display's refresh rate,
  presents each frame on its exact refresh and reports measured pacing.
  Video only for now.

Not yet: audio in the player, the real UI (Flutter, needs Xcode), Apple's
VideoToolbox frame-rate conversion (needs macOS 15.4+ and its SDK), AI
engines, colour pipeline, network and link playback.

## Layout

```
core/include/advp/   public C++ API (engine, rate, governor, bench, device, convert)
core/src/            portable logic + FFmpeg converter
core/src/apple/      Metal backend and macOS device probe
core/shaders/        Metal kernels (compiled at runtime)
cli/                 advp developer CLI
tests/               unit tests and the quality benchmark
tools/               FFmpeg build script and wrapper
```

## Build (macOS)

Needs only the Xcode Command Line Tools for now.

```bash
python3 -m venv .venv && .venv/bin/pip install cmake ninja
```

```bash
./tools/build_ffmpeg.sh
```

```bash
.venv/bin/cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=$PWD/.venv/bin/ninja && .venv/bin/cmake --build build
```

## Use

```bash
build/advp probe
```

```bash
build/advp bench --size 1920x1080
```

```bash
build/advp convert movie.mkv movie_60.mp4 --fps 60
```

```bash
build/advp play movie.mkv
```

Player keys: space pause, ←/→ seek 10 s, S smoothing on/off (before/after),
F full screen, Q or Esc quit. `--engine off|blend|motion-fast|motion` fixes the
engine; `--seconds N` stops after N seconds and prints pacing stats;
`--screenshot frame.png` saves one smoothed frame as displayed.

Output rate (`--fps`):

| Value | Frames made | When to use |
|---|---|---|
| `display` (default) | one per screen refresh | smoothest; most GPU work |
| `fit:60` | highest rate up to 60 that divides the refresh evenly (50 on 100 Hz, 60 on 60/120 Hz) | steady and lighter; leaves room for a better engine on heavy video |
| `60`, `2x` | exactly that rate | uneven hold if it does not divide the refresh (the player says so) |

With a fixed rate each frame is made once, ahead of time, while the previous
one is on screen.

`--fps` takes `60`, `120`, `59.94` or `2x`; an integer keeps NTSC timing
(23.976 → 59.94). `--engine` takes `auto`, `blend`, `motion-fast` or `motion`.

## Tests

```bash
build/advp_tests
```

```bash
tests/quality.sh
```

`quality.sh` renders 60 fps reference clips, drops every other frame, rebuilds
them with each engine and reports PSNR of the rebuilt frames.

```bash
tests/film_quality.sh samples/film.mkv
```

`film_quality.sh` does the same on real footage: it drops every other frame
of a 24 fps film (24 → 12), rebuilds 24 fps and scores the rebuilt frames,
leaving out pairs that span a scene cut. Each rebuilt frame spans twice the
motion that 24 → 48/60 playback sees, so its scores are lower than real use;
use it to compare engines and builds, and look at the worst frames it lists.
