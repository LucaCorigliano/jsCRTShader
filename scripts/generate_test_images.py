#!/usr/bin/env python3
"""
Generate reference CRT test patterns and write them to assets/test-images.
Run this when the palette of built-in images needs to be refreshed.
"""

from __future__ import annotations

import math
import os
import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "assets" / "test-images"


def write_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data))
            + tag
            + data
            + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack("!IIBBBBB", width, height, 8, 6, 0, 0, 0)
    raw = bytearray()
    stride = width * 4
    for y in range(height):
        raw.append(0)  # filter type 0
        raw.extend(pixels[y * stride : (y + 1) * stride])
    data = zlib.compress(bytes(raw), 9)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", data) + chunk(b"IEND", b"")
    path.write_bytes(png)


def make_canvas(width: int, height: int, rgb: tuple[int, int, int]) -> bytearray:
    r, g, b = rgb
    return bytearray([r, g, b, 255] * (width * height))


def fill_rect(buf: bytearray, width: int, height: int, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    r, g, b = color
    x0 = max(0, min(width, x0))
    y0 = max(0, min(height, y0))
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    for y in range(y0, y1):
        idx = (y * width + x0) * 4
        for _ in range(x1 - x0):
            buf[idx] = r
            buf[idx + 1] = g
            buf[idx + 2] = b
            idx += 4


def smpte_pattern(width: int, height: int) -> bytearray:
    buf = make_canvas(width, height, (16, 16, 16))
    top_colors = [
        (191, 191, 191),
        (191, 191, 16),
        (16, 191, 191),
        (16, 191, 16),
        (191, 16, 191),
        (191, 16, 16),
        (16, 16, 191),
    ]
    mid_colors = [
        (16, 16, 191),
        (16, 16, 16),
        (191, 16, 191),
        (16, 16, 16),
        (16, 191, 191),
        (16, 16, 16),
        (96, 96, 96),
    ]
    top_h = int(height * 0.67)
    mid_h = int(height * 0.11)
    x = 0
    for i, c in enumerate(top_colors):
        nxt = int(round((i + 1) * width / len(top_colors)))
        fill_rect(buf, width, height, x, 0, nxt, top_h, c)
        x = nxt
    x = 0
    for i, c in enumerate(mid_colors):
        nxt = int(round((i + 1) * width / len(mid_colors)))
        fill_rect(buf, width, height, x, top_h, nxt, top_h + mid_h, c)
        x = nxt
    bottom_y = top_h + mid_h
    bottom_h = height - bottom_y
    pluge_w = int(width * 0.18)
    fill_rect(buf, width, height, 0, bottom_y, pluge_w, height, (16, 16, 16))
    center = pluge_w // 2
    bar = max(3, pluge_w // 12)
    fill_rect(buf, width, height, center - bar * 2, bottom_y, center - bar, height, (5, 5, 5))
    fill_rect(buf, width, height, center - bar, bottom_y, center, height, (16, 16, 16))
    fill_rect(buf, width, height, center, bottom_y, center + bar, height, (32, 32, 32))
    i_w = int(width * 0.22)
    q_w = int(width * 0.22)
    fill_rect(buf, width, height, pluge_w, bottom_y, pluge_w + i_w, height, (0, 180, 180))
    fill_rect(buf, width, height, pluge_w + i_w, bottom_y, pluge_w + i_w + q_w, height, (180, 0, 180))
    gray_x0 = pluge_w + i_w + q_w
    steps = 6
    step_w = max(1, (width - gray_x0) // steps)
    for i in range(steps):
        val = int((i / (steps - 1)) * 219) + 16
        fill_rect(buf, width, height, gray_x0 + i * step_w, bottom_y, gray_x0 + (i + 1) * step_w, height, (val, val, val))
    return buf


def ebu_pattern(width: int, height: int) -> bytearray:
    buf = make_canvas(width, height, (16, 16, 16))
    top_colors = [
        (235, 235, 235),
        (235, 235, 16),
        (16, 235, 235),
        (16, 235, 16),
        (235, 16, 235),
        (235, 16, 16),
        (16, 16, 235),
        (16, 16, 16),
    ]
    top_h = int(height * 0.72)
    x = 0
    for i, c in enumerate(top_colors):
        nxt = int(round((i + 1) * width / len(top_colors)))
        fill_rect(buf, width, height, x, 0, nxt, top_h, c)
        x = nxt
    chroma_y = top_h
    chroma_h = int(height * 0.14)
    chroma_colors = [
        (16, 16, 16),
        (16, 16, 235),
        (16, 235, 16),
        (235, 16, 16),
        (16, 235, 235),
        (235, 235, 16),
        (235, 235, 235),
        (16, 16, 16),
    ]
    x = 0
    for i, c in enumerate(chroma_colors):
        nxt = int(round((i + 1) * width / len(chroma_colors)))
        fill_rect(buf, width, height, x, chroma_y, nxt, chroma_y + chroma_h, c)
        x = nxt
    gray_y = chroma_y + chroma_h
    gray_steps = 12
    gray_w = max(1, width // gray_steps)
    for i in range(gray_steps):
        val = int((i / (gray_steps - 1)) * 219) + 16
        fill_rect(buf, width, height, i * gray_w, gray_y, (i + 1) * gray_w, height, (val, val, val))
    return buf


def rec709_geometry(width: int, height: int) -> bytearray:
    buf = make_canvas(width, height, (19, 19, 19))
    for y in range(height):
        t = y / max(1, height - 1)
        val = int(16 + 219 * t)
        row = y * width * 4
        for x in range(width):
            idx = row + x * 4
            buf[idx] = val
            buf[idx + 1] = val
            buf[idx + 2] = val
    bar_h = int(height * 0.2)
    bar_colors = [
        (235, 235, 235),
        (235, 235, 16),
        (16, 235, 235),
        (16, 235, 16),
        (235, 16, 235),
        (235, 16, 16),
        (16, 16, 235),
    ]
    x = 0
    for i, c in enumerate(bar_colors):
        nxt = int(round((i + 1) * width / len(bar_colors)))
        fill_rect(buf, width, height, x, 0, nxt, bar_h, c)
        x = nxt
    cx = width / 2.0
    cy = height / 2.0
    def draw_circle(radius: float, color: tuple[int, int, int]) -> None:
        r = int(radius)
        x1 = r
        y1 = 0
        decision = 1 - r
        while y1 <= x1:
            points = [
                (cx + x1, cy + y1),
                (cx + y1, cy + x1),
                (cx - x1, cy + y1),
                (cx - y1, cy + x1),
                (cx - x1, cy - y1),
                (cx - y1, cy - x1),
                (cx + x1, cy - y1),
                (cx + y1, cy - x1),
            ]
            for px, py in points:
                ix = int(round(px))
                iy = int(round(py))
                if 0 <= ix < width and 0 <= iy < height:
                    idx = (iy * width + ix) * 4
                    buf[idx], buf[idx + 1], buf[idx + 2] = color
            y1 += 1
            if decision <= 0:
                decision += 2 * y1 + 1
            else:
                x1 -= 1
                decision += 2 * (y1 - x1) + 1
    draw_circle(min(width, height) * 0.4, (235, 235, 235))
    draw_circle(min(width, height) * 0.25, (16, 16, 16))
    for y in range(0, height, max(4, height // 32)):
        fill_rect(buf, width, height, 0, y, width, min(height, y + 1), (48, 48, 48))
    for x in range(0, width, max(4, width // 32)):
        fill_rect(buf, width, height, x, 0, min(width, x + 1), height, (48, 48, 48))
    fill_rect(buf, width, height, int(cx) - 1, 0, int(cx) + 1, height, (235, 235, 235))
    fill_rect(buf, width, height, 0, int(cy) - 1, width, int(cy) + 1, (235, 235, 235))
    return buf


def crosshatch(width: int, height: int) -> bytearray:
    buf = make_canvas(width, height, (0, 0, 0))
    major = max(1, width // 80)
    for x in range(0, width):
        color = (180, 180, 180)
        if x % (major * 5) == 0:
            color = (235, 235, 235)
        fill_rect(buf, width, height, x, 0, x + 1, height, color)
    for y in range(0, height):
        color = (180, 180, 180)
        if y % (major * 5) == 0:
            color = (235, 235, 235)
        fill_rect(buf, width, height, 0, y, width, y + 1, color)
    border = 4
    fill_rect(buf, width, height, 0, 0, width, border, (235, 235, 235))
    fill_rect(buf, width, height, 0, height - border, width, height, (235, 235, 235))
    fill_rect(buf, width, height, 0, 0, border, height, (235, 235, 235))
    fill_rect(buf, width, height, width - border, 0, width, height, (235, 235, 235))
    return buf


def generate() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUTPUT_DIR.glob("*.png"):
        old.unlink()
    patterns = [
        ("smpte-ntsc-320x240.png", smpte_pattern, 320, 240),
        ("smpte-ntsc-640x480.png", smpte_pattern, 640, 480),
        ("ebu-pal-720x576.png", ebu_pattern, 720, 576),
        ("ebu-pal-768x576.png", ebu_pattern, 768, 576),
        ("crosshatch-640x480.png", crosshatch, 640, 480),
        ("crosshatch-1280x720.png", crosshatch, 1280, 720),
        ("rec709-geometry-1920x1080.png", rec709_geometry, 1920, 1080),
    ]
    for filename, generator, width, height in patterns:
        pixels = generator(width, height)
        write_png(OUTPUT_DIR / filename, width, height, pixels)
    print(f"Generated {len(patterns)} test images in {OUTPUT_DIR}")


if __name__ == "__main__":
    generate()
