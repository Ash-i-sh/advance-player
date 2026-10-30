#!/usr/bin/env bash
# Interpolation quality on real footage. Real film has no true in-between
# frames, so every other frame is dropped (24 -> 12 fps), 24 fps is rebuilt
# with each engine, and the rebuilt frames are compared with the dropped ones
# (PSNR, higher is better). Motion per rebuilt frame is twice what 24 -> 48/60
# playback sees, so scores are lower than real use; compare engines and builds.
#
#   tests/film_quality.sh FILM [workdir]
# Env: SIZE (output width, default 1920), CHUNK (seconds per pass, default 20),
#      SECONDS_MAX (stop early), ADVP_BIN (compare another build).
# Pairs that span a scene cut are left out of the scores (the engines repeat a
# frame there by design). Cuts come from ffmpeg's scdet, which misses some; at
# a missed cut the dropped frame belongs to one shot or the other, so an engine
# that rightly repeats a frame still scores badly half the time. Per-frame
# scores land in WORK/frames.csv; look at the frames before trusting a number.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FILM="$1"
WORK="${2:-$ROOT/build/film}"
FF="$ROOT/tools/ff"
ADVP="${ADVP_BIN:-$ROOT/build/advp}"
SIZE="${SIZE:-1920}"
CHUNK="${CHUNK:-20}"
ENGINES=(blend motion-fast motion)
mkdir -p "$WORK"

fps=$("$FF" ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of csv=p=0 "$FILM")
duration=$("$FF" ffprobe -v error -show_entries format=duration -of csv=p=0 "$FILM")
read -r frames chunk_frames fps <<<"$(python3 -I -c "
from fractions import Fraction as F
fps = F('$fps'); limit = float('${SECONDS_MAX:-$duration}')
print(int(min(float('$duration'), limit) * fps), int('$CHUNK') * round(fps), float(fps))")"
half_fps=$(python3 -I -c "from fractions import Fraction as F; print(F('$fps') / 2)")

# Scene cuts, as the index of the first frame of each new shot.
if [ ! -s "$WORK/cuts.txt" ]; then
  "$FF" ffmpeg -loglevel error -i "$FILM" -an \
    -vf "scale=480:-2,scdet=threshold=12,metadata=print:key=lavfi.scd.time:file=$WORK/cuts.txt" \
    -f null -
fi

: >"$WORK/frames.csv.tmp"
for ((first = 0; first < frames; first += chunk_frames)); do
  last=$((first + chunk_frames < frames ? first + chunk_frames : frames))
  ref="trim=start_frame=$first:end_frame=$last,setpts=PTS-STARTPTS,scale=$SIZE:-2:flags=bicubic"
  half="$WORK/half.mkv"
  "$FF" ffmpeg -y -loglevel error -i "$FILM" -an \
    -vf "$ref,select='not(mod(n\,2))',setpts=N/($half_fps)/TB" -r "$half_fps" -c:v ffv1 "$half"
  printf "\r  frames %d-%d of %d   " "$first" "$last" "$frames" >&2

  inputs=(-i "$FILM" -i "$half")
  graph="[0:v]$ref,settb=1/60,setpts=N,split=$((${#ENGINES[@]} + 1))"
  for ((i = 0; i <= ${#ENGINES[@]}; i++)); do graph+="[r$i]"; done
  graph+=";[1:v]fps=$fps,settb=1/60,setpts=N[t0];[t0][r0]psnr=stats_file=$WORK/repeat.log"
  for ((i = 0; i < ${#ENGINES[@]}; i++)); do
    out="$WORK/${ENGINES[$i]}.mp4"
    "$ADVP" convert "$half" "$out" --fps "$fps" --engine "${ENGINES[$i]}" --quality 95 >/dev/null 2>&1
    inputs+=(-i "$out")
    graph+=";[$((i + 2)):v]settb=1/60,setpts=N[t$((i + 1))];[t$((i + 1))][r$((i + 1))]"
    graph+="psnr=stats_file=$WORK/${ENGINES[$i]}.log"
  done
  "$FF" ffmpeg -loglevel error "${inputs[@]}" -lavfi "$graph" -f null -
  python3 -I - "$WORK" "$first" "$((last - first))" repeat "${ENGINES[@]}" >>"$WORK/frames.csv.tmp" <<'EOF'
import re, sys
work, first, count, names = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
scores = []
for name in names:
    per = {}
    for line in open(f"{work}/{name}.log"):
        n = int(re.search(r"n:(\d+)", line).group(1)) - 1
        v = re.search(r"psnr_avg:(\S+)", line).group(1)
        per[n] = 100.0 if v == "inf" else float(v)
    scores.append(per)
# Odd frames were dropped and rebuilt; the chunk's last frame has no later
# neighbour to rebuild from.
for n in range(1, count - 1, 2):
    if all(n in s for s in scores):
        print(first + n, *(f"{s[n]:.3f}" for s in scores), sep=",")
EOF
done
echo >&2
rm -f "$WORK/half.mkv"
{ echo "frame,repeat,$(IFS=,; echo "${ENGINES[*]}")"; cat "$WORK/frames.csv.tmp"; } >"$WORK/frames.csv"
rm -f "$WORK/frames.csv.tmp"

python3 -I - "$WORK" <<'EOF'
import csv, re, sys
work = sys.argv[1]
cuts = {int(m.group(1)) for m in re.finditer(r"frame:(\d+)", open(f"{work}/cuts.txt").read())}
rows = list(csv.DictReader(open(f"{work}/frames.csv")))
names = [k for k in rows[0] if k != "frame"]
# A rebuilt frame n sits between n-1 and n+1; a cut at n or n+1 splits them.
kept = [r for r in rows if int(r["frame"]) not in cuts and int(r["frame"]) + 1 not in cuts]
print(f"{len(kept)} rebuilt frames scored ({len(rows) - len(kept)} at scene cuts left out)")
for name in names:
    v = sorted(float(r[name]) for r in kept)
    mean = sum(v) / len(v)
    print(f"  {name:12s} {mean:6.2f} dB (worst 10% {v[len(v) // 10]:5.2f}, worst {v[0]:5.2f})")
# Where motion loses most against the plain blend: the frames to look at.
worst = sorted(kept, key=lambda r: float(r["motion"]) - float(r["blend"]))[:8]
print("motion vs blend, worst frames:",
      ", ".join(f"{r['frame']} ({float(r['motion']) - float(r['blend']):+.1f})" for r in worst))
EOF
