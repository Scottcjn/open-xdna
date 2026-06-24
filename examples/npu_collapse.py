#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Scott Boudreaux / Elyan Labs. Commercial license: see ../COMMERCIAL.md
#
# open-xdna :: non-bijunctive COLLAPSE on the gen-1 XDNA1 (Phoenix) NPU.
#
# Composes two PROVEN NPU kernels into a real top-k-style prune (pse-vcipher-collapse on silicon):
#   1) reduce_max  (NPU)  -> peak score M           [proven: vector_reduce_max single_core]
#   2) relu        (NPU)  -> keep strong, drop weak  [proven: eltwise_unary -o relu / P0]
#
# Collapse rule:  survivors = relu(scores - tau),  tau = KEEP_FRACTION-derived threshold = f * M.
# The peak reduction and the elementwise collapse BOTH run on the NPU. The scalar (tau = f*M)
# and the shift are host-side (trivial O(1)/O(N) metadata — and per Grok's review the selection
# decision is cheap; the point of this file is to MEASURE the real collapse, not to claim 3x).
#
# Run (root or render-group; llvm-objcopy on PATH):
#   export PATH=/usr/lib/llvm-20/bin:$PATH
#   source ~/open-xdna/mlir-aie/ironenv/bin/activate
#   source /opt/xilinx/xrt/setup.sh; source ~/open-xdna/mlir-aie/utils/env_setup.sh
#   python3 examples/npu_collapse.py

import sys, time
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import In, Out, CompileTime, kernels
from aie.iron.algorithms import reduce, transform_parallel

N = 8192            # score-vector length
KEEP_FRAC = 0.50    # keep scores >= KEEP_FRAC * peak (tune for prune ratio)


@iron.jit
def npu_reduce_max(a_in: In, c_out: Out, *, num_elements: CompileTime[int]):
    in_ty = np.ndarray[(num_elements,), np.dtype[bfloat16]]
    out_ty = np.ndarray[(2,), np.dtype[bfloat16]]   # 4-byte slot; bf16 fills 2, [0] is the max
    return reduce(kernels.reduce_max(tile_size=num_elements, dtype=bfloat16), in_ty, out_ty)


@iron.jit
def npu_relu(a_in: In, b_out: Out, *, size: CompileTime[int]):
    return transform_parallel(
        kernels.relu(tile_size=1024),
        np.ndarray[(size,), np.dtype[bfloat16]],
        tile_size=1024, num_channels=2, pass_size_to_kernel=False,
    )


def main():
    rng = np.random.default_rng(0)
    scores = rng.uniform(-3.0, 3.0, size=(N,)).astype(bfloat16)

    print(f"open-xdna :: non-bijunctive collapse on XDNA1 NPU  (N={N}, bf16)")

    # 1) NPU reduce_max -> peak
    a_t = iron.tensor(scores, dtype=bfloat16, device="npu")
    m_t = iron.tensor(np.zeros(2, dtype=bfloat16), dtype=bfloat16, device="npu")
    t0 = time.perf_counter()
    npu_reduce_max(a_t, m_t, num_elements=N)
    t_rmax = (time.perf_counter() - t0) * 1e3
    peak = float(m_t.numpy()[0])
    tau = KEEP_FRAC * peak
    print(f"  [NPU] reduce_max -> peak={peak:.3f}  ({t_rmax:.1f} ms)   tau={tau:.3f} (keep>= {KEEP_FRAC}*peak)")

    # 2) NPU relu( scores - tau ) -> collapse (survivors are > 0)
    shifted = (scores.astype(np.float32) - tau).astype(bfloat16)   # host shift (cheap metadata)
    s_t = iron.tensor(shifted, dtype=bfloat16, device="npu")
    o_t = iron.zeros_like(s_t)
    t0 = time.perf_counter()
    npu_relu(s_t, o_t, size=N)
    t_relu = (time.perf_counter() - t0) * 1e3
    collapsed = o_t.numpy().astype(np.float32)

    survivors = int(np.count_nonzero(collapsed > 0))
    prune_ratio = 1.0 - survivors / N
    # host reference
    ref = np.maximum(scores.astype(np.float32) - tau, 0.0)
    ok = np.allclose(collapsed, ref, atol=0.05)

    print(f"  [NPU] relu collapse  ({t_relu:.1f} ms)")
    print(f"  survivors={survivors}/{N}  -> PRUNED {prune_ratio*100:.1f}%   (kept top {100-prune_ratio*100:.1f}%)")
    print(f"  total NPU collapse latency: {t_rmax + t_relu:.1f} ms")
    print(f"  matches host reference: {ok}")
    print("  RESULT:", "PASS — non-bijunctive collapse ran on the NPU" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
