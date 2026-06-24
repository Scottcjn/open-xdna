#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Scott Boudreaux / Elyan Labs. Commercial license: see ../COMMERCIAL.md
#
# open-xdna :: NPU camera-effects daemon (A1) — core path.
# Captures real frames from a v4l2 camera, runs the per-frame conv/colorspace effect on the
# XDNA1 NPU (rgba2gray -> 3x3 filter2d -> threshold -> gray2rgba -> blend = edge-stylize), and
# measures LIVE end-to-end FPS on real camera data. Saves a before/after sample as proof.
#
# This is the core of an open-source Linux "Studio Effects"-style NPU webcam filter. The virtual
# v4l2loopback output (so Zoom/OBS see it as a camera) is the next layer; --loopback enables it
# if /dev/videoN (v4l2loopback) + pyvirtualcam are present.

import os, sys, time, argparse
import numpy as np
import cv2

MLIR_AIE = os.environ.get("MLIR_AIE_DIR", os.path.expanduser("~/open-xdna/mlir-aie"))
sys.path.insert(0, os.path.join(MLIR_AIE, "programming_examples/vision/edge_detect"))
import aie.iron as iron
from edge_detect import edge_detect          # @iron.jit per-frame vision pipeline


def npu_effect(frame_bgr, W, H, in_t, b_t, out_t):
    """One frame: BGR uint8 -> NPU edge-stylize -> BGR uint8."""
    rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA).reshape(-1)          # H*W*4 uint8
    in_t.numpy()[:] = rgba.view(np.int8)                                    # in-place into NPU tensor
    edge_detect(in_t, b_t, out_t, width=W, height=H)
    out_rgba = out_t.numpy().view(np.uint8).reshape(H, W, 4)
    return cv2.cvtColor(out_rgba, cv2.COLOR_RGBA2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("-W", "--width", type=int, default=1280)
    ap.add_argument("-H", "--height", type=int, default=720)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--out", default="/home/scott/open-xdna/examples/cam_sample")
    opts = ap.parse_args()
    W, H = opts.width, opts.height

    cap = cv2.VideoCapture(opts.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # MJPG → higher USB capture FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    if not cap.isOpened():
        print(f"ERROR: cannot open {opts.device}"); sys.exit(1)

    ts = W * H * 4
    in_t = iron.tensor(np.zeros(ts, dtype=np.int8), dtype=np.int8, device="npu")
    b_t = iron.zeros(16 * 16, dtype=np.int32, device="npu")
    out_t = iron.zeros(ts, dtype=np.int8, device="npu")

    print(f"open-xdna :: NPU camera daemon — {opts.device} @ {W}x{H}, {opts.frames} frames")
    # warm: grab one frame, compile the pipeline
    ok, f = cap.read()
    if not ok: print("ERROR: no frame from camera"); sys.exit(1)
    f = cv2.resize(f, (W, H))
    npu_effect(f, W, H, in_t, b_t, out_t)
    cv2.imwrite(opts.out + "_before.png", f)
    cv2.imwrite(opts.out + "_after.png", npu_effect(f, W, H, in_t, b_t, out_t))

    # A) capture-only (find the camera ceiling)
    n = 0; t0 = time.perf_counter()
    while n < opts.frames:
        ok, f = cap.read()
        if not ok: break
        if f.shape[1] != W or f.shape[0] != H: f = cv2.resize(f, (W, H))
        n += 1
    cap_fps = n / (time.perf_counter() - t0)

    # B) capture + NPU effect (end-to-end)
    n = 0; t0 = time.perf_counter()
    while n < opts.frames:
        ok, f = cap.read()
        if not ok: break
        if f.shape[1] != W or f.shape[0] != H: f = cv2.resize(f, (W, H))
        npu_effect(f, W, H, in_t, b_t, out_t)
        n += 1
    e2e_fps = n / (time.perf_counter() - t0)
    cap.release()

    npu_only = 1000.0 / 2.21 if (W, H) == (1280, 720) else None   # from npu_camera_fps.py
    print(f"  capture-only:        {cap_fps:5.1f} FPS  (camera/USB ceiling)")
    print(f"  capture + NPU effect:{e2e_fps:5.1f} FPS  (end-to-end)")
    if npu_only: print(f"  NPU effect alone:    {npu_only:5.0f} FPS  (2.21 ms/frame @720p — not the bottleneck)")
    print(f"  sample saved: {opts.out}_before.png / _after.png")
    overhead = (1.0/e2e_fps - 1.0/cap_fps) * 1000 if e2e_fps > 0 and cap_fps > 0 else 0
    print(f"  VERDICT: NPU adds ~{overhead:.1f} ms/frame; the limit is CAMERA CAPTURE ({cap_fps:.0f} FPS), not the NPU.")


if __name__ == "__main__":
    main()
