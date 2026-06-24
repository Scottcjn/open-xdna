#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Scott Boudreaux / Elyan Labs. Commercial license: see ../COMMERCIAL.md
#
# open-xdna :: run the HAND-AUTHORED AIE-intrinsic collapse kernel (collapse.cc) on XDNA1.
# Proves you can write your own AIE vector kernels (compare+select, the pse-vcipher core) and
# dispatch them through IRON, exactly like the built-in kernels — AltiVec instinct, AIE ISA.
#
# Kernel source: mlir_aie/include/aie_kernels/aie2/collapse.cc  (also examples/kernels/collapse.cc)
#
# Run (root or render-group; llvm-objcopy on PATH; env_setup sourced).

import sys, time
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import In, Out, CompileTime
from aie.iron.algorithms import transform_parallel
from aie.iron.kernels._common import _make_extern, _default_source_path
from aie.utils.verify import assert_pass

N = 8192
TAU = 0.5  # must match the baked tau in collapse.cc

# Auto-install the kernel source where IRON's _default_source_path expects it (the kernel uses
# a relative include "../aie_kernel_utils.h", so it must live in the aie_kernels/aie2 dir).
import os, shutil
_DST = str(_default_source_path("collapse.cc"))
_SRC = os.path.join(os.path.dirname(__file__), "kernels", "collapse.cc")
if os.path.exists(_SRC) and not os.path.exists(_DST):
    shutil.copy(_SRC, _DST)
    print(f"installed collapse.cc -> {_DST}")


@iron.jit
def npu_collapse_hw(a_in: In, b_out: Out, *, size: CompileTime[int] = N):
    tile_ty = np.ndarray[(1024,), np.dtype[bfloat16]]
    collapse_k = _make_extern("bf16_collapse", _default_source_path("collapse.cc"),
                              [tile_ty, tile_ty])
    return transform_parallel(
        collapse_k,
        np.ndarray[(size,), np.dtype[bfloat16]],
        tile_size=1024, num_channels=2, pass_size_to_kernel=False,
    )


def main():
    rng = np.random.default_rng(0)
    scores = rng.uniform(-3.0, 3.0, size=(N,)).astype(bfloat16)
    a_t = iron.tensor(scores, dtype=bfloat16, device="npu")
    b_t = iron.zeros_like(a_t)

    print(f"open-xdna :: HAND-AUTHORED AIE-intrinsic collapse kernel on XDNA1  (N={N}, tau={TAU})")
    t0 = time.perf_counter()
    npu_collapse_hw(a_t, b_t, size=N)
    dt = (time.perf_counter() - t0) * 1e3

    out = b_t.numpy().astype(np.float32)
    ref = np.where(scores.astype(np.float32) >= TAU, scores.astype(np.float32), 0.0)
    survivors = int(np.count_nonzero(out != 0))
    ok = np.allclose(out, ref, atol=0.05)
    print(f"  [NPU] collapse.cc (aie::ge + aie::select)  ({dt:.0f} ms incl JIT compile)")
    print(f"  survivors={survivors}/{N}  -> PRUNED {100*(1-survivors/N):.1f}%")
    print(f"  matches host reference (x>=tau?x:0): {ok}")
    print("  RESULT:", "PASS — custom AIE collapse kernel ran on the NPU" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
