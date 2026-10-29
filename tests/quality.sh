#!/usr/bin/env bash
# Objective interpolation quality: render 60 fps reference content, drop every
# other frame, rebuild 60 fps with each engine, and compare the rebuilt frames
# with the real ones (PSNR, higher is better).
#
#   tests/quality.sh [workdir]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${1:-$ROOT/build/quality}"
FF="$ROOT/tools/ff"
ADVP="$ROOT/build/advp"
mkdir -p "$WORK"

# A detailed still image under a camera-like pan (speed varies, motion is
# rarely a whole number of pixels), plus a test card with moving objects.
"$FF" ffmpeg -y -loglevel error -f lavfi -i "mandelbrot=size=2880x1620:rate=1" -frames:v 1 \
  "$WORK/still.png"
"$FF" ffmpeg -y -loglevel error -loop 1 -i "$WORK/still.png" \
  -vf "fps=60,crop=1920:1080:x='480+420*sin(t*1.1)':y='270+250*cos(t*0.9)',format=yuv420p" \
  -t 4 -c:v ffv1 "$WORK/pan_60.mkv"
"$FF" ffmpeg -y -loglevel error -f lavfi -i "testsrc2=size=1920x1080:rate=60,format=yuv420p" \
  -t 4 -c:v ffv1 "$WORK/card_60.mkv"

# Odd-numbered (interpolated) frames vs even-numbered (kept) frames.
psnr_report() {
  python3 -I - "$1" <<'EOF'
import re, sys
odd, even = [], []
for line in open(sys.argv[1]):
    n = int(re.search(r"n:(\d+)", line).group(1))
    v = re.search(r"psnr_avg:(\S+)", line).group(1)
    v = 100.0 if v == "inf" else float(v)
    (odd if n % 2 == 0 else even).append(v)  # stats count from 1
mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
print(f"rebuilt frames {mean(odd):6.2f} dB   kept frames {mean(even):6.2f} dB")
EOF
}

for clip in pan card; do
  ref="$WORK/${clip}_60.mkv"
  half="$WORK/${clip}_30.mkv"
  "$FF" ffmpeg -y -loglevel error -i "$ref" \
    -vf "select='not(mod(n\,2))',setpts=N/30/TB" -r 30 -c:v ffv1 "$half"
  echo "== $clip =="

  # Baseline: show each kept frame twice (what a normal player does).
  "$FF" ffmpeg -y -loglevel error -i "$half" -vf "fps=60" -c:v ffv1 "$WORK/${clip}_repeat.mkv"
  "$FF" ffmpeg -loglevel error -i "$WORK/${clip}_repeat.mkv" -i "$ref" \
    -lavfi "[0:v]setpts=N/60/TB[a];[1:v]setpts=N/60/TB[b];[a][b]psnr=stats_file=$WORK/${clip}_repeat.log" -f null -
  printf "  %-12s " repeat; psnr_report "$WORK/${clip}_repeat.log"

  for engine in blend motion-fast motion; do
    out="$WORK/${clip}_${engine}.mp4"
    "$ADVP" convert "$half" "$out" --fps 60 --engine "$engine" --quality 95 >/dev/null 2>&1
    "$FF" ffmpeg -loglevel error -i "$out" -i "$ref" \
      -lavfi "[0:v]setpts=N/60/TB[a];[1:v]setpts=N/60/TB[b];[a][b]psnr=stats_file=$WORK/${clip}_${engine}.log" -f null -
    printf "  %-12s " "$engine"; psnr_report "$WORK/${clip}_${engine}.log"
  done
done
