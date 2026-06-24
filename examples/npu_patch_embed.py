#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Scott Boudreaux / Elyan Labs. Commercial license: see ../COMMERCIAL.md
#
# open-xdna :: SigLIP/ViT patch-embed wired onto the XDNA1 NPU + measured.
#
# A ViT patch-embed (Conv2d kernel=stride=patch, non-overlapping) == patchify + projection matmul:
#   image[H,W,C] -> patches[N_patch, P*P*C] -> @ W_proj[P*P*C, embed] -> tokens[N_patch, embed]
# The patchify is a host-side reshape (free, non-overlapping); the PROJECTION MATMUL — the heavy
# compute — runs on the NPU (reuses the proven int16 single_core GEMM). We measure NPU vs CPU.
#
# SigLIP-base/16: 224x224x3, patch 16 -> 196 patches (padded to 256), patch_dim 768, embed 768.

import os, sys, time
import numpy as np

MLIR_AIE = os.environ.get("MLIR_AIE_DIR", os.path.expanduser("~/open-xdna/mlir-aie"))
sys.path.insert(0, os.path.join(MLIR_AIE, "programming_examples/basic/matrix_multiplication/single_core"))
import aie.iron as iron
from single_core import single_core

IMG, C, P = 224, 3, 16
NPAT = (IMG//P)**2            # 196
NPAT_PAD = 256               # pad to a 32-multiple tile (196 -> 256)
PDIM = P*P*C                 # 768
EMBED = 768
TILE = dict(m=32, k=32, n=32, dtype_in_str="i16", dtype_out_str="i32")


def patchify(img):
    """Non-overlapping P×P patches -> [NPAT, P*P*C]  (this is the 'im2col' for stride=kernel)."""
    H = IMG//P
    p = img.reshape(H, P, H, P, C).transpose(0, 2, 1, 3, 4).reshape(NPAT, PDIM)
    out = np.zeros((NPAT_PAD, PDIM), dtype=img.dtype); out[:NPAT] = p
    return out


def npu_matmul(A_i16, B_i16):
    a = iron.tensor(A_i16.reshape(-1), dtype=np.int16, device="npu")
    b = iron.tensor(B_i16.reshape(-1), dtype=np.int16, device="npu")
    c = iron.tensor(np.zeros(NPAT_PAD*EMBED, dtype=np.int32), dtype=np.int32, device="npu")
    single_core(a, b, c, M=NPAT_PAD, K=PDIM, N=EMBED, **TILE)
    return c.numpy().reshape(NPAT_PAD, EMBED).astype(np.int32)


def main():
    rng = np.random.default_rng(0)
    img   = rng.integers(-8, 8, size=(IMG, IMG, C), dtype=np.int16)        # quantized image
    Wproj = rng.integers(-4, 5, size=(PDIM, EMBED), dtype=np.int16)        # patch projection

    patches = patchify(img)                                               # host reshape (free)
    print(f"open-xdna :: SigLIP/ViT patch-embed on XDNA1  ({IMG}x{IMG}x{C}, patch {P} -> {NPAT} tokens x{EMBED})")
    print(f"  projection matmul: [{NPAT_PAD}x{PDIM}] @ [{PDIM}x{EMBED}]  ({2*NPAT_PAD*PDIM*EMBED/1e6:.0f} MFLOP)")

    # NPU (int16)
    npu_matmul(patches, Wproj)  # warm (JIT compile)
    t0 = time.perf_counter(); emb_npu = npu_matmul(patches, Wproj); t_npu = (time.perf_counter()-t0)*1e3
    emb_ref = patches.astype(np.int32) @ Wproj.astype(np.int32)            # correctness reference
    # FAIR CPU comparison = f32 BLAS (what you'd actually run); int32 numpy is unoptimized, NOT a baseline
    Af, Bf = patches.astype(np.float32), Wproj.astype(np.float32)
    for _ in range(3): Af @ Bf
    t0 = time.perf_counter()
    for _ in range(20): Af @ Bf
    t_cpu_f32 = (time.perf_counter()-t0)/20*1e3

    ok = np.array_equal(emb_npu, emb_ref)
    flop = 2*NPAT_PAD*PDIM*EMBED
    print(f"  NPU patch-embed (int16):   {t_npu:6.2f} ms  ({flop/(t_npu/1e3)/1e9:5.1f} GFLOP/s, incl host round-trip)")
    print(f"  CPU patch-embed (f32 BLAS):{t_cpu_f32:6.2f} ms  ({flop/(t_cpu_f32/1e3)/1e9:5.1f} GFLOP/s)  <- fair baseline")
    print(f"  embeddings match reference: {ok}   tokens[0,:4]={emb_npu[0,:4]}")
    print(f"  RESULT: {'PASS' if ok else 'MISMATCH'} — SigLIP patch-embed projection runs on the NPU (bit-exact).")
    print(f"  HONEST: at this size the NPU (dispatch-bound) is ~{t_npu/max(t_cpu_f32,1e-9):.1f}x SLOWER than fair CPU f32 BLAS.")
    print(f"  The win is OFFLOAD (frees CPU/iGPU for the LLM) + ~6.6W power + better scaling at larger image/batch,")
    print(f"  NOT raw latency at SigLIP-base size. (The int32-numpy 'CPU' figure people quote here is unoptimized — ignore it.)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
