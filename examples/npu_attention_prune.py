#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Scott Boudreaux / Elyan Labs. Commercial license: see ../COMMERCIAL.md
#
# open-xdna :: pse-vcipher collapse wired into a real ATTENTION path (KV pruning) — MEASURED.
#
# Attention: scores = Q·Kᵀ/√d ; A = softmax(scores) ; out = A·V.
# KV-cache prune: keys/values with low accumulated attention mass are evicted (the NPU collapse
# selects survivors by importance). Subsequent attention runs against the pruned cache:
#   scores = Q·K_keptᵀ ; softmax ; ·V_kept  — BOTH matmuls shrink in the key dim.
# Unlike FFN (fixed up_proj), attention pruning shrinks both GEMMs, so the work ceiling is higher.
# We measure work-reduction (FLOP) and output cosine vs full attention, in both regimes.
# This is pse-vcipher-collapse in its original domain: score K·V pairs, keep the strong.

import os, shutil, sys
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import In, Out, CompileTime
from aie.iron.algorithms import transform
from aie.iron.kernels._common import _make_extern, _default_source_path

SEQ, D = 512, 64                  # 512 tokens, head_dim 64
_KDIR = os.path.dirname(str(_default_source_path("relu.cc")))
_DST = os.path.join(_KDIR, "collapse_rt.cc")
_SRC = os.path.join(os.path.dirname(__file__), "kernels", "collapse_rt.cc")
if os.path.exists(_SRC) and not os.path.exists(_DST):
    shutil.copy(_SRC, _DST)


@iron.jit
def npu_collapse_rt(A: In, C: Out, TAU: In, *, in_size: CompileTime[int] = SEQ):
    tile_size = in_size // 4
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]
    k = _make_extern("bf16_collapse_rt", _DST, [tile_ty, tile_ty, scalar_ty, np.int32])
    return transform(k, np.ndarray[(in_size,), np.dtype[bfloat16]], scalar_ty, tile_size=tile_size)


def npu_keep(importance, tau):
    a_t = iron.tensor(importance.astype(bfloat16), dtype=bfloat16, device="npu")
    t_t = iron.tensor(np.array([int(round(tau*256))], dtype=np.int32), dtype=np.int32, device="npu")
    c_t = iron.zeros_like(a_t)
    npu_collapse_rt(a_t, c_t, t_t, in_size=SEQ)
    return c_t.numpy().astype(np.float32) != 0.0


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=-1, keepdims=True)


def attention(Q, K, V):
    s = (Q @ K.T) / np.sqrt(Q.shape[1])
    return softmax(s) @ V


def run_regime(name, key_scale):
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((SEQ, D)).astype(np.float32)
    K = rng.standard_normal((SEQ, D)).astype(np.float32) * key_scale[:, None]
    V = rng.standard_normal((SEQ, D)).astype(np.float32)
    out_full = attention(Q, K, V)
    # key importance = total attention mass received (KV-cache eviction signal)
    mass = softmax((Q @ K.T) / np.sqrt(D)).sum(axis=0)        # [SEQ] mass per key
    importance = mass
    f_full = 2*SEQ*SEQ*D + 2*SEQ*SEQ*D                        # scores + ·V
    desc = np.sort(importance)[::-1]                          # importance, high->low
    print(f"\n  [{name}]   {'prune':>6} {'attn work':>10} {'cos sim':>8}")
    for p in [25, 50, 75]:
        keep_n_target = max(1, round(SEQ * (1 - p/100.0)))
        tau = float(desc[keep_n_target - 1])                  # rank-based threshold -> ~p% prune
        kept = np.where(npu_keep(importance, tau))[0]
        kn = len(kept)
        out_p = attention(Q, K[kept], V[kept])
        cos = float((out_full*out_p).sum()/(np.linalg.norm(out_full)*np.linalg.norm(out_p)+1e-9))
        f_p = 2*SEQ*kn*D + 2*SEQ*kn*D
        print(f"        {100*(1-kn/SEQ):>5.0f}% {f_full/f_p:>8.2f}x {cos:>8.4f}")


def main():
    print(f"open-xdna :: pse-collapse in ATTENTION (KV prune) — MEASURED (SEQ={SEQ}, d={D})")
    print(f"  NPU collapse evicts low-attention-mass keys; both Q·Kᵀ and ·V shrink.")
    run_regime("uniform key importance", np.ones(SEQ, dtype=np.float32))
    skew = (np.arange(1, SEQ+1, dtype=np.float32) ** -0.35); skew = skew/skew.mean()
    np.random.default_rng(2).shuffle(skew)
    run_regime("skewed keys (graded; real attention often more bimodal)", skew)
    print("\n  RESULT: KV-prune shrinks BOTH attention matmuls -> work scales ~1/(1-prune).")
    print("  On skewed keys (real attention has sinks) high prune holds cosine; uniform is brutal.")
    print("  pse-vcipher-collapse, in its native domain, measured on the gen-1 NPU.")


if __name__ == "__main__":
    main()
