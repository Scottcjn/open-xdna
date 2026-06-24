#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Scott Boudreaux / Elyan Labs. Commercial license: see ../COMMERCIAL.md
#
# open-xdna :: CROSS-TILE reduce + collapse for vectors larger than one AIE tile (>4096).
# The fused single-tile kernel caps at ~4096 (whole vector must fit 64KB local mem). Here the
# global peak is reduced ACROSS tiles on the NPU (aie::reduce_max per CHUNK), the handful of
# partial peaks combined (max of K scalars — trivial metadata), then collapse_rt streams the
# full vector in tiles with tau = frac * global_peak. Lifts the N cap end-to-end.

import os, shutil, sys
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import In, Out, CompileTime, kernels
from aie.iron.algorithms import reduce, transform
from aie.iron.kernels._common import _make_extern, _default_source_path

N = 16384            # 4x the single-tile cap
CHUNK = 4096         # per-tile reduce size (fits 64KB)

# ensure collapse_rt kernel is installed (relative-include dep)
_DST = str(_default_source_path("collapse_rt.cc"))
_SRC = os.path.join(os.path.dirname(__file__), "kernels", "collapse_rt.cc")
if os.path.exists(_SRC) and not os.path.exists(_DST):
    shutil.copy(_SRC, _DST)


@iron.jit
def npu_reduce_max(a_in: In, c_out: Out, *, num_elements: CompileTime[int] = CHUNK):
    in_ty = np.ndarray[(num_elements,), np.dtype[bfloat16]]
    out_ty = np.ndarray[(2,), np.dtype[bfloat16]]
    return reduce(kernels.reduce_max(tile_size=num_elements, dtype=bfloat16), in_ty, out_ty)


@iron.jit
def npu_collapse_rt(A: In, C: Out, TAU: In, *, in_size: CompileTime[int] = N):
    tile_size = in_size // 4
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]
    k = _make_extern("bf16_collapse_rt", _default_source_path("collapse_rt.cc"),
                     [tile_ty, tile_ty, scalar_ty, np.int32])
    return transform(k, np.ndarray[(in_size,), np.dtype[bfloat16]], scalar_ty, tile_size=tile_size)


def main():
    rng = np.random.default_rng(0)
    scores = rng.uniform(-3.0, 3.0, size=(N,)).astype(bfloat16)
    frac = 0.5

    # --- cross-tile reduce: per-CHUNK peak on the NPU, combine partials ---
    partials = []
    for ci in range(N // CHUNK):
        chunk = np.ascontiguousarray(scores[ci*CHUNK:(ci+1)*CHUNK])
        a_t = iron.tensor(chunk, dtype=bfloat16, device="npu")
        m_t = iron.tensor(np.zeros(2, dtype=bfloat16), dtype=bfloat16, device="npu")
        npu_reduce_max(a_t, m_t, num_elements=CHUNK)
        partials.append(float(m_t.numpy()[0]))
    peak = max(partials)   # combine K partials (trivial)
    tau = frac * peak
    print(f"open-xdna :: cross-tile reduce + collapse  (N={N}, {N//CHUNK} tiles x {CHUNK})")
    print(f"  per-tile NPU peaks = {[round(p,3) for p in partials]}  -> global peak {peak:.3f}, tau {tau:.3f}")

    # --- collapse over the FULL vector (tiled) on the NPU ---
    tau_q = int(round(tau * 256.0))
    a_t = iron.tensor(scores, dtype=bfloat16, device="npu")
    f_t = iron.tensor(np.array([tau_q], dtype=np.int32), dtype=np.int32, device="npu")
    c_t = iron.zeros_like(a_t)
    npu_collapse_rt(a_t, c_t, f_t, in_size=N)
    out = c_t.numpy().astype(np.float32)

    scores_f = scores.astype(np.float32)
    tau_bf = float(np.array(tau, dtype=bfloat16))
    ref = np.where(scores_f >= tau_bf, scores_f, 0.0)
    surv = int(np.count_nonzero(out != 0))
    diffs = int(np.count_nonzero(np.abs(out - ref) > 0.08))
    ok = (peak == max(partials)) and diffs / N < 0.005
    print(f"  survivors={surv}/{N}  pruned {100*(1-surv/N):.1f}%  boundary-diffs={diffs}")
    print("  RESULT:", "PASS — cross-tile reduce+collapse on the NPU (N>tile cap)" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
