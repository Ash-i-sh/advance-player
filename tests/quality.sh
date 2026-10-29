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
ADVP="${ADVP_BIN:-$ROOT/build/advp}"  # ADVP_BIN: compare another build
mkdir -p "$WORK"

# A detailed still image under a smooth sub-pixel camera pan (speed varies,
# rendered exactly by make_pan.py), plus a test card with moving objects.
# Reference clips are reused between runs; delete $WORK to rebuild them.
if [ ! -f "$WORK/pan_60.mkv" ]; then
  "$FF" ffmpeg -y -loglevel error -f lavfi -i "mandelbrot=size=2880x1620:rate=1" -frames:v 1 \
    "$WORK/still.png"
  "$FF" ffmpeg -y -loglevel error -i "$WORK/still.png" -f rawvideo -pix_fmt rgb24 "$WORK/still.rgb"
  python3 -I "$ROOT/tests/make_pan.py" "$WORK/still.rgb" 2880 1620 240 60 |
    "$FF" ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgb24 -s 1920x1080 -r 60 -i - \
      -vf format=yuv420p -c:v ffv1 "$WORK/pan_60.mkv"
  "$FF" ffmpeg -y -loglevel error -f lavfi -i "testsrc2=size=1920x1080:rate=60,format=yuv420p" \
    -t 4 -c:v ffv1 "$WORK/card_60.mkv"
fi

# Odd-numbered (interpolated) frames vs even-numbered (kept) frames.
psnr_report() {
  python3 -I - "$1" <<'EOF'
import re, sys
odd, even = [], []
for line in open(sys.argv[1]):
    n = int(re.search(r"n:(\d+)", line).group(1))
    v = re.search(r"psnr_avg:(\S+)", line).group(1)
    v = 100.0 if v == "inf" else float(v)
    if n >= 240:
        continue  # the source's last frame is held at the end; nothing to rebuild
    (odd if n % 2 == 0 else even).append(v)  # stats count from 1
mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
odd.sort()
p10 = odd[len(odd) // 10] if odd else float("nan")
worst = odd[0] if odd else float("nan")
print(f"rebuilt {mean(odd):6.2f} dB (worst 10% {p10:5.2f}, worst {worst:5.2f})"
      f"   kept {mean(even):6.2f} dB")
EOF
}

for clip in ${CLIPS:-pan card}; do  # CLIPS: run a subset
  ref="$WORK/${clip}_60.mkv"
  half="$WORK/${clip}_30.mkv"
  "$FF" ffmpeg -y -loglevel error -i "$ref" \
    -vf "select='not(mod(n\,2))',setpts=N/30/TB" -r 30 -c:v ffv1 "$half"
  echo "== $clip =="

  # Baseline: show each kept frame twice (what a normal player does).
  "$FF" ffmpeg -y -loglevel error -i "$half" -vf "fps=60" -c:v ffv1 "$WORK/${clip}_repeat.mkv"
  "$FF" ffmpeg -loglevel error -i "$WORK/${clip}_repeat.mkv" -i "$ref" \
    -lavfi "[0:v]settb=1/60,setpts=N[a];[1:v]settb=1/60,setpts=N[b];[a][b]psnr=stats_file=$WORK/${clip}_repeat.log" -f null -
  printf "  %-12s " repeat; psnr_report "$WORK/${clip}_repeat.log"

  for engine in blend motion-fast motion; do
    out="$WORK/${clip}_${engine}.mp4"
    "$ADVP" convert "$half" "$out" --fps 60 --engine "$engine" --quality 95 >/dev/null 2>&1
    "$FF" ffmpeg -loglevel error -i "$out" -i "$ref" \
      -lavfi "[0:v]settb=1/60,setpts=N[a];[1:v]settb=1/60,setpts=N[b];[a][b]psnr=stats_file=$WORK/${clip}_${engine}.log" -f null -
    printf "  %-12s " "$engine"; psnr_report "$WORK/${clip}_${engine}.log"
  done
done
