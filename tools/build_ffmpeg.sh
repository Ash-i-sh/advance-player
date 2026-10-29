#!/usr/bin/env bash
# Builds an LGPL-only FFmpeg into third_party/ffmpeg.
# We never pass --enable-gpl or --enable-nonfree: the product is sold, so every
# FFmpeg component we ship must stay LGPL.
set -euo pipefail

FFMPEG_TAG="${FFMPEG_TAG:-n8.1.3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/third_party/src/ffmpeg-$FFMPEG_TAG"
PREFIX="$ROOT/third_party/ffmpeg"
JOBS="$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

if [ ! -d "$SRC" ]; then
  mkdir -p "$ROOT/third_party/src"
  git clone --depth 1 --branch "$FFMPEG_TAG" https://git.ffmpeg.org/ffmpeg.git "$SRC"
fi

cd "$SRC"

COMMON=(
  --prefix="$PREFIX"
  --enable-shared --disable-static
  --disable-doc --disable-debug
  --disable-autodetect          # never pick up random Homebrew libs
  --enable-zlib --enable-bzlib --enable-iconv
)

case "$(uname -s)" in
  Darwin)
    PLATFORM=(
      --install-name-dir='@rpath'
      --enable-videotoolbox --enable-audiotoolbox
      --enable-securetransport  # https for network playback
      --extra-libs=-liconv      # libiconv is separate from libSystem on macOS
    )
    ;;
  *)
    PLATFORM=()
    ;;
esac

./configure "${COMMON[@]}" "${PLATFORM[@]}"

# Fail loudly if configure somehow switched to a GPL/nonfree build.
if grep -qE '^#define CONFIG_(GPL|NONFREE) 1' config.h; then
  echo "FFmpeg configured as GPL/nonfree — refusing to build" >&2
  exit 1
fi

make -j"$JOBS"
make install
echo "FFmpeg $FFMPEG_TAG (LGPL) installed to $PREFIX"
