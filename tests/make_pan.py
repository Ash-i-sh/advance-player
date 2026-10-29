"""Renders a smooth sub-pixel camera pan over a still image as raw RGB frames.

ffmpeg's crop filter moves in whole pixels, so its in-between frames are off
by up to half a pixel from a real pan and cap the measured quality. Here each
frame is shifted exactly (Fourier phase shift), like a real camera would see.

    python3 -I make_pan.py STILL.rgb WIDTH HEIGHT FRAMES FPS > frames.rgb
STILL.rgb is raw rgb24 at least 512 pixels larger than 1920x1080 each way.
"""
import math
import sys

import numpy as np

OUT_W, OUT_H = 1920, 1080
PAD = 64  # margin around the crop so the periodic FFT shift has no wrap-around


def main():
    path, w, h, frames, fps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
    still = np.fromfile(path, np.uint8).reshape(h, w, 3).astype(np.float32)
    cw, ch = OUT_W + 2 * PAD, OUT_H + 2 * PAD
    fx = np.fft.fftfreq(cw)[None, :, None]
    fy = np.fft.fftfreq(ch)[:, None, None]
    out = sys.stdout.buffer
    for n in range(frames):
        t = n / fps
        # Varying speed, rarely a whole number of pixels per frame.
        x = (w - OUT_W) / 2 + 200 * math.sin(t * 1.1)
        y = (h - OUT_H) / 2 + 120 * math.cos(t * 0.9)
        ix, iy = int(math.floor(x)), int(math.floor(y))
        sx, sy = x - ix, y - iy
        crop = still[iy - PAD:iy - PAD + ch, ix - PAD:ix - PAD + cw]
        spec = np.fft.fft2(crop, axes=(0, 1))
        spec *= np.exp(2j * math.pi * (fx * sx + fy * sy))
        frame = np.fft.ifft2(spec, axes=(0, 1)).real[PAD:PAD + OUT_H, PAD:PAD + OUT_W]
        out.write(np.clip(frame + 0.5, 0, 255).astype(np.uint8).tobytes())


if __name__ == "__main__":
    main()
