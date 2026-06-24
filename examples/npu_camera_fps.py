#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Scott Boudreaux / Elyan Labs. Commercial license: see ../COMMERCIAL.md
#
# open-xdna :: FPS gate for an NPU camera-effects daemon (A1).
# Times the full per-frame conv/colorspace pipeline (rgba2gray -> 3x3 filter2d -> threshold ->
# gray2rgba -> blend) on the XDNA1 NPU at camera resolutions, INCLUDING the host DMA round-trip
# (fill input frame -> run -> drain output) — exactly what a v4l2 daemon pays per frame.
# Gate: >=30 FPS at 720p/1080p means real-time camera effects on the NPU are feasible.

import os, sys, time
import numpy as np

MLIR_AIE = os.environ.get("MLIR_AIE_DIR", os.path.expanduser("~/open-xdna/mlir-aie"))
sys.path.insert(0, os.path.join(MLIR_AIE, "programming_examples/vision/edge_detect"))
import aie.iron as iron
from edge_detect import edge_detect          # the @iron.jit per-frame vision pipeline


def bench(W, H, iters=50):
    ts = W * H * 4                            # RGBA int8 frame
    rng = np.random.default_rng(0)
    in_t = iron.tensor(rng.integers(-128, 127, size=(ts,), dtype=np.int8), dtype=np.int8, device="npu")
    b_t = iron.zeros(16 * 16, dtype=np.int32, device="npu")
    out_t = iron.zeros(ts, dtype=np.int8, device="npu")
    # warm/compile (JIT specializes per W,H)
    edge_detect(in_t, b_t, out_t, width=W, height=H)
    edge_detect(in_t, b_t, out_t, width=W, height=H)
    t0 = time.perf_counter()
    for _ in range(iters):
        edge_detect(in_t, b_t, out_t, width=W, height=H)
    dt = (time.perf_counter() - t0) / iters
    return dt * 1e3, 1.0 / dt


def main():
    print("open-xdna :: NPU camera-effects FPS gate (full conv/colorspace pipeline + DMA round-trip)")
    print(f"  {'res':>8} {'pixels':>10} {'ms/frame':>10} {'FPS':>8}   verdict")
    for W, H, name in [(640, 480, "480p"), (1280, 720, "720p"), (1920, 1080, "1080p")]:
        ms, fps = bench(W, H)
        v = "OK ≥60" if fps >= 60 else ("OK ≥30" if fps >= 30 else "below 30 — too slow")
        print(f"  {name:>8} {W*H:>10} {ms:>10.2f} {fps:>8.1f}   {v}")
    print("  (per-frame = full pipeline + host fill/drain; this is the daemon's real per-frame cost)")


if __name__ == "__main__":
    main()
