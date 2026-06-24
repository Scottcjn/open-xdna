#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Scott Boudreaux / Elyan Labs. Commercial license: see ../COMMERCIAL.md
#
# open-xdna :: prove the vec_perm / lane-permute primitive runs on XDNA1 (shuffle_demo.cc).
# Reverses each 32-lane vector via aie::reverse — the building block for top-k survivor
# compaction. (The full data-dependent dense pack = prefix-sum + scatter, the next frontier;
# this verifies the AIE permute engine itself.)

import os, shutil, sys
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import In, Out, CompileTime
from aie.iron.algorithms import transform_parallel
from aie.iron.kernels._common import _make_extern, _default_source_path

N = 8192
_DST = str(_default_source_path("shuffle_demo.cc"))
_SRC = os.path.join(os.path.dirname(__file__), "kernels", "shuffle_demo.cc")
if os.path.exists(_SRC) and not os.path.exists(_DST):
    shutil.copy(_SRC, _DST); print(f"installed shuffle_demo.cc -> {_DST}")


@iron.jit
def npu_shuffle(A: In, C: Out, *, size: CompileTime[int] = N):
    tile_ty = np.ndarray[(1024,), np.dtype[bfloat16]]
    k = _make_extern("bf16_shuffle_demo", _default_source_path("shuffle_demo.cc"),
                     [tile_ty, tile_ty])
    return transform_parallel(k, np.ndarray[(size,), np.dtype[bfloat16]],
                              tile_size=1024, num_channels=2, pass_size_to_kernel=False)


def main():
    rng = np.random.default_rng(0)
    x = rng.uniform(-3.0, 3.0, size=(N,)).astype(bfloat16)
    a = iron.tensor(x, dtype=bfloat16, device="npu"); c = iron.zeros_like(a)
    npu_shuffle(a, c, size=N)
    out = c.numpy().astype(np.float32)
    ref = x.astype(np.float32).reshape(-1, 32)[:, ::-1].reshape(-1)  # per-32 reverse
    ok = np.allclose(out, ref, atol=0.05)
    print(f"open-xdna :: aie::reverse (vec_perm) on XDNA1, N={N}")
    print(f"  matches per-32-lane reverse: {ok}")
    print("  RESULT:", "PASS — vec_perm/shuffle runs on the NPU" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
