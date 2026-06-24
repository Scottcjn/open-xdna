#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Scott Boudreaux / Elyan Labs. Commercial license: see ../COMMERCIAL.md
#
# open-xdna :: runtime-tau non-bijunctive collapse on XDNA1 (collapse_rt.cc).
# ONE compiled AIE kernel; the threshold tau is supplied at RUNTIME (Q8 int32 scalar) and the
# threshold test is computed on the NPU via aie::sub (the AltiVec vec_sub equivalent):
#   x - tau >= 0  <=>  x >= tau.   Sweeps several tau without recompiling.

import os, shutil, sys
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import In, Out, CompileTime
from aie.iron.algorithms import transform
from aie.iron.kernels._common import _make_extern, _default_source_path

N = 8192

# auto-install the kernel where IRON expects it (relative-include dependency)
_DST = str(_default_source_path("collapse_rt.cc"))
_SRC = os.path.join(os.path.dirname(__file__), "kernels", "collapse_rt.cc")
if os.path.exists(_SRC) and not os.path.exists(_DST):
    shutil.copy(_SRC, _DST)
    print(f"installed collapse_rt.cc -> {_DST}")


@iron.jit
def npu_collapse_rt(A: In, C: Out, TAU: In, *, in_size: CompileTime[int] = N):
    tile_size = in_size // 4
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]
    coll = _make_extern("bf16_collapse_rt", _default_source_path("collapse_rt.cc"),
                        [tile_ty, tile_ty, scalar_ty, np.int32])
    tensor_ty = np.ndarray[(in_size,), np.dtype[bfloat16]]
    return transform(coll, tensor_ty, scalar_ty, tile_size=tile_size)


def run_tau(scores, tau):
    tau_q = int(round(tau * 256.0))   # Q8 fixed-point
    a_t = iron.tensor(scores, dtype=bfloat16, device="npu")
    tau_t = iron.tensor(np.array([tau_q], dtype=np.int32), dtype=np.int32, device="npu")
    c_t = iron.zeros_like(a_t)
    npu_collapse_rt(a_t, c_t, tau_t, in_size=len(scores))
    out = c_t.numpy().astype(np.float32)
    ref = np.where(scores.astype(np.float32) >= tau, scores.astype(np.float32), 0.0)
    surv = int(np.count_nonzero(out != 0))
    return surv, np.allclose(out, ref, atol=0.06)


def main():
    rng = np.random.default_rng(0)
    scores = rng.uniform(-3.0, 3.0, size=(N,)).astype(bfloat16)
    print(f"open-xdna :: runtime-tau collapse on XDNA1 (N={N}; one kernel, tau at runtime via aie::sub)")
    allok = True
    for tau in [0.0, 0.5, 1.5, 2.5]:
        surv, ok = run_tau(scores, tau)
        allok &= ok
        print(f"  tau={tau:>4.1f}: survivors={surv:5d}/{N}  pruned {100*(1-surv/N):5.1f}%  match={ok}")
    print("  RESULT:", "PASS — runtime-tau collapse on the NPU" if allok else "MISMATCH")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
