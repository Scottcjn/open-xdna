#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Scott Boudreaux / Elyan Labs. Commercial license: see ../COMMERCIAL.md
#
# open-xdna :: FUSED reduce_max -> dynamic-tau collapse on XDNA1 (collapse_fused.cc).
# One AIE kernel finds the data peak (aie::reduce_max), sets tau = frac*peak in-kernel, and
# collapses (keep x >= tau). Only `frac` is supplied at runtime — tau adapts to the data.

import os, shutil, sys
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import In, Out, CompileTime
from aie.iron.algorithms import transform
from aie.iron.kernels._common import _make_extern, _default_source_path

N = 4096
_DST = str(_default_source_path("collapse_fused.cc"))
_SRC = os.path.join(os.path.dirname(__file__), "kernels", "collapse_fused.cc")
if os.path.exists(_SRC) and not os.path.exists(_DST):
    shutil.copy(_SRC, _DST); print(f"installed collapse_fused.cc -> {_DST}")


@iron.jit
def npu_collapse_fused(A: In, C: Out, FRAC: In, *, in_size: CompileTime[int] = N):
    # single-tile (no sub-split) so reduce_max sees the whole vector in one kernel call
    tile_ty = np.ndarray[(in_size,), np.dtype[bfloat16]]
    scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]
    k = _make_extern("bf16_collapse_fused", _default_source_path("collapse_fused.cc"),
                     [tile_ty, tile_ty, scalar_ty, np.int32])
    tensor_ty = np.ndarray[(in_size,), np.dtype[bfloat16]]
    return transform(k, tensor_ty, scalar_ty, tile_size=in_size)


def run_frac(scores, frac):
    frac_q = int(round(frac * 256.0))
    a_t = iron.tensor(scores, dtype=bfloat16, device="npu")
    f_t = iron.tensor(np.array([frac_q], dtype=np.int32), dtype=np.int32, device="npu")
    c_t = iron.zeros_like(a_t)
    npu_collapse_fused(a_t, c_t, f_t, in_size=len(scores))
    out = c_t.numpy().astype(np.float32)
    # reference computed the SAME way the kernel does (bf16-rounded tau) so the comparison is faithful
    scores_f = scores.astype(np.float32)
    peak = float(scores_f.max())
    frac_eff = frac_q / 256.0
    tau = float(np.array(frac_eff * peak, dtype=bfloat16))   # bf16-rounded, as in-kernel
    ref = np.where(scores_f >= tau, scores_f, 0.0)
    surv = int(np.count_nonzero(out != 0))
    diffs = int(np.count_nonzero(np.abs(out - ref) > 0.08))
    ok = diffs / len(scores) < 0.005   # allow <0.5% bf16-ULP boundary straddle
    return surv, tau, diffs, ok


def main():
    rng = np.random.default_rng(0)
    scores = rng.uniform(-3.0, 3.0, size=(N,)).astype(bfloat16)
    print(f"open-xdna :: FUSED reduce_max->collapse on XDNA1 (N={N}, peak found in-kernel)")
    allok = True
    for frac in [0.0, 0.5, 0.83]:
        surv, tau, diffs, ok = run_frac(scores, frac)
        allok &= ok
        print(f"  frac={frac:>4.2f}  tau=frac*peak~={tau:5.2f}  survivors={surv:5d}/{N}  pruned {100*(1-surv/N):5.1f}%  boundary-diffs={diffs}  match={ok}")
    print("  RESULT:", "PASS — fused reduce_max+collapse on the NPU" if allok else "MISMATCH")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
