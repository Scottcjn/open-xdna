#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Scott Boudreaux / Elyan Labs. Commercial license: see ../COMMERCIAL.md
#
# open-xdna :: PROOF-OF-CONCEPT — a tiny model's forward pass running its matmuls on the
# first-generation AMD XDNA1 (Phoenix / Hawk Point) NPU.
#
# This is the "it actually runs a model" demo. There is no full XDNA ggml backend; instead
# we compose the PROVEN NPU matmul kernel (IRON single_core, int16x int16->int32, 512^3 —
# the shape verified at ~68 GFLOPS) into a 2-layer MLP:
#
#     h = relu(x @ W1)        # matmul #1 on the NPU
#     y = h @ W2              # matmul #2 on the NPU
#
# Both 512x512x512 matmuls EXECUTE ON THE NPU. The relu is applied host-side here (the relu
# kernel also runs on the NPU — see ml/eltwise_unary -o relu, proven separately in P0), and
# the int32->int16 requantization between layers is host-side (a real low-precision MLP).
#
# Run (as root or render-group, with llvm-objcopy on PATH):
#   export PATH=/usr/lib/llvm-20/bin:$PATH
#   source ~/open-xdna/mlir-aie/ironenv/bin/activate
#   source /opt/xilinx/xrt/setup.sh; source ~/open-xdna/mlir-aie/utils/env_setup.sh
#   python3 examples/npu_tiny_mlp.py

import os, sys
import numpy as np

MLIR_AIE = os.environ.get("MLIR_AIE_DIR", os.path.expanduser("~/open-xdna/mlir-aie"))
SC_DIR = os.path.join(MLIR_AIE, "programming_examples/basic/matrix_multiplication/single_core")
sys.path.insert(0, SC_DIR)

import aie.iron as iron
from single_core import single_core  # the proven @iron.jit 512^3 int16 matmul

D = 512  # model width == proven NPU matmul tile shape (M=K=N=512)
TILE = dict(m=32, k=32, n=32, dtype_in_str="i16", dtype_out_str="i32")


def npu_matmul(a_i16, b_i16):
    """a @ b on the NPU. a,b int16 [512,512] -> int32 [512,512]."""
    a_t = iron.tensor(a_i16.reshape(-1), dtype=np.int16, device="npu")
    b_t = iron.tensor(b_i16.reshape(-1), dtype=np.int16, device="npu")
    c_t = iron.tensor(np.zeros(D * D, dtype=np.int32), dtype=np.int32, device="npu")
    single_core(a_t, b_t, c_t, M=D, K=D, N=D, **TILE)
    return c_t.numpy().reshape(D, D).astype(np.int32)


def requant_i16(x_i32):
    """relu + scale back into int16 range (host-side, between NPU layers)."""
    x = np.maximum(x_i32, 0)                       # relu  (also an NPU kernel; see P0)
    m = x.max()
    if m > 32767:
        x = (x.astype(np.float64) * (32767.0 / m)).astype(np.int16)
    return x.astype(np.int16)


def main():
    rng = np.random.default_rng(0)
    x  = rng.integers(-4, 5, size=(D, D), dtype=np.int16)   # one 512-token activation block
    W1 = rng.integers(-4, 5, size=(D, D), dtype=np.int16)
    W2 = rng.integers(-4, 5, size=(D, D), dtype=np.int16)

    print(f"open-xdna :: tiny MLP forward pass on XDNA1 NPU  (width {D}, int16)")
    print("  layer 1: h = relu(x @ W1)   [matmul on NPU]")
    h_pre = npu_matmul(x, W1)
    h = requant_i16(h_pre)
    print("  layer 2: y = h @ W2         [matmul on NPU]")
    y_npu = npu_matmul(h, W2)

    # host reference using the SAME quantization path
    h_ref = requant_i16((x.astype(np.int32) @ W1.astype(np.int32)))
    y_ref = h_ref.astype(np.int32) @ W2.astype(np.int32)

    ok = np.array_equal(y_npu, y_ref)
    err = np.abs(y_npu - y_ref).max()
    print(f"\n  2 matmuls executed on gen-1 Phoenix NPU.")
    print(f"  output[0,:4] = {y_npu[0,:4]}")
    print(f"  matches host reference: {ok}  (max abs diff {err})")
    print("  RESULT:", "PASS — tiny model ran on the NPU" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
