#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Scott Boudreaux / Elyan Labs. Commercial license: see ../COMMERCIAL.md
#
# open-xdna :: full stream COMPACTION on XDNA1 (compact.cc) — dense survivor pack.
# Survivors (x >= tau) are packed contiguously at the front, tail zero-filled — the top-k pack
# that lets a downstream matmul run on a dense [0:count] slice. Scalar-unit compaction (correct
# baseline); vectorized prefix-sum + shuffle-gather is the perf optimization.

import os, shutil, sys
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import In, Out, CompileTime
from aie.iron.algorithms import transform
from aie.iron.kernels._common import _make_extern, _default_source_path

N = 4096   # single-tile (≤64KB local mem); >N via cross-tile partial-count prefix (see xtile)
TAU = 1.5
# anchor on an existing kernel's dir (compact.cc isn't there until we copy it)
_KDIR = os.path.dirname(str(_default_source_path("relu.cc")))
_DST = os.path.join(_KDIR, "compact.cc")
_SRC = os.path.join(os.path.dirname(__file__), "kernels", "compact.cc")
if os.path.exists(_SRC) and not os.path.exists(_DST):
    shutil.copy(_SRC, _DST); print(f"installed compact.cc -> {_DST}")


@iron.jit
def npu_compact(A: In, C: Out, TAU_: In, *, in_size: CompileTime[int] = N):
    tile_ty = np.ndarray[(in_size,), np.dtype[bfloat16]]
    scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]
    k = _make_extern("bf16_compact", _DST, [tile_ty, tile_ty, scalar_ty, np.int32])
    return transform(k, np.ndarray[(in_size,), np.dtype[bfloat16]], scalar_ty, tile_size=in_size)


def main():
    rng = np.random.default_rng(0)
    scores = rng.uniform(-3.0, 3.0, size=(N,)).astype(bfloat16)
    tau_q = int(round(TAU * 256.0))
    a_t = iron.tensor(scores, dtype=bfloat16, device="npu")
    t_t = iron.tensor(np.array([tau_q], dtype=np.int32), dtype=np.int32, device="npu")
    c_t = iron.zeros_like(a_t)
    npu_compact(a_t, c_t, t_t, in_size=N)
    out = c_t.numpy().astype(np.float32)

    sf = scores.astype(np.float32)
    tau = float(np.array(tau_q / 256.0, dtype=bfloat16))
    survivors = sf[sf >= tau]                      # host reference: dense survivors
    ref = np.zeros(N, dtype=np.float32); ref[:len(survivors)] = survivors
    count = len(survivors)
    out_count = int(np.count_nonzero(out != 0))
    dense_ok = np.allclose(out[:count], ref[:count], atol=0.06)
    packed_ok = np.all(out[count:] == 0)           # tail is zeros (densely packed at front)
    print(f"open-xdna :: stream COMPACTION on XDNA1 (N={N}, tau={TAU})")
    print(f"  survivors packed: {count}/{N}  (out nonzeros={out_count})")
    print(f"  first 6 packed   : {out[:6]}")
    print(f"  dense-front matches survivors: {dense_ok}   tail all-zero: {packed_ok}")
    ok = dense_ok and packed_ok and out_count == count
    print("  RESULT:", "PASS — dense survivor compaction on the NPU" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
