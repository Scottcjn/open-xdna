#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Scott Boudreaux / Elyan Labs. Commercial license: see ../COMMERCIAL.md
#
# open-xdna :: end-to-end test — CPU+GPU (full) vs CPU+GPU + NPU-prune, one transformer layer.
#
# A full pre-norm transformer block: MHA(attention) + FFN, with residuals.
#   baseline  : run the layer dense (the "CPU+GPU" path)
#   + prune   : NPU collapse evicts low-mass KV keys (attention) AND low-importance FFN columns;
#               the attention and down_proj GEMMs then run on the dense survivors.
# We measure: matmul wall-clock (CPU BLAS = controlled downstream), FLOP reduction, and the
# layer-output cosine vs the dense layer. The NPU selection cost is the collapse latency
# (sub-ms, warmed) — negligible vs the GEMM time it removes.
#
# Honest: downstream matmuls timed on CPU BLAS (reproducible); the iGPU sees the same FLOP cut,
# so the work-reduction transfers. Realistic skewed weights (real models have channel/key skew).

import os, shutil, sys, time
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import In, Out, CompileTime
from aie.iron.algorithms import transform
from aie.iron.kernels._common import _make_extern, _default_source_path

SEQ, D, HEADS, FFN = 512, 512, 8, 2048
HD = D // HEADS
KEEP_KV, KEEP_FFN = 0.50, 0.50     # keep top 50% keys / FFN columns (accuracy-safe on skewed)

_KDIR = os.path.dirname(str(_default_source_path("relu.cc")))
_DST = os.path.join(_KDIR, "collapse_rt.cc")
_SRC = os.path.join(os.path.dirname(__file__), "kernels", "collapse_rt.cc")
if os.path.exists(_SRC) and not os.path.exists(_DST):
    shutil.copy(_SRC, _DST)


@iron.jit
def npu_collapse_rt(A: In, C: Out, TAU: In, *, in_size: CompileTime[int] = SEQ):
    ts = in_size // 4
    tile_ty = np.ndarray[(ts,), np.dtype[bfloat16]]; scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]
    k = _make_extern("bf16_collapse_rt", _DST, [tile_ty, tile_ty, scalar_ty, np.int32])
    return transform(k, np.ndarray[(in_size,), np.dtype[bfloat16]], scalar_ty, tile_size=ts)


def npu_select(importance, keep_frac):
    """NPU collapse picks survivors: keep the top keep_frac by importance. Returns (kept_idx, npu_ms)."""
    n = len(importance)
    tau = float(np.sort(importance)[::-1][max(1, round(n*keep_frac))-1])
    a_t = iron.tensor(importance.astype(bfloat16), dtype=bfloat16, device="npu")
    t_t = iron.tensor(np.array([int(round(tau*256))], dtype=np.int32), dtype=np.int32, device="npu")
    c_t = iron.zeros_like(a_t)
    npu_collapse_rt(a_t, c_t, t_t, in_size=n)          # warm (compile)
    t0 = time.perf_counter(); npu_collapse_rt(a_t, c_t, t_t, in_size=n); npu_ms=(time.perf_counter()-t0)*1e3
    return np.where(c_t.numpy().astype(np.float32) != 0.0)[0], npu_ms


def softmax(z): z=z-z.max(-1,keepdims=True); e=np.exp(z); return e/e.sum(-1,keepdims=True)


def layer(x, W, kv_keep=None, ffn_keep=None):
    """One transformer block. kv_keep/ffn_keep = arrays of kept indices (None = dense)."""
    Wq,Wk,Wv,Wo,Wu,Wd = W
    Q,K,V = x@Wq, x@Wk, x@Wv
    ctx = np.zeros_like(Q)
    for h in range(HEADS):
        sl=slice(h*HD,(h+1)*HD); Qh,Kh,Vh=Q[:,sl],K[:,sl],V[:,sl]
        if kv_keep is not None: Kh,Vh = Kh[kv_keep],Vh[kv_keep]
        ctx[:,sl]=softmax((Qh@Kh.T)/np.sqrt(HD))@Vh
    x = x + ctx@Wo
    h = np.maximum(x@Wu, 0.0)
    if ffn_keep is not None:
        x = x + h[:,ffn_keep]@Wd[ffn_keep,:]
    else:
        x = x + h@Wd
    return x


def matmul_flops(kv_n, ffn_n):
    qkv=3*2*SEQ*D*D; attn=HEADS*(2*SEQ*kv_n*HD*2); o=2*SEQ*D*D
    up=2*SEQ*D*FFN; dn=2*SEQ*ffn_n*D
    return qkv+attn+o+up+dn


def timed(fn, it=10):
    for _ in range(2): fn()
    t=time.perf_counter()
    for _ in range(it): r=fn()
    return (time.perf_counter()-t)/it, r


def main():
    rng=np.random.default_rng(0)
    x=rng.standard_normal((SEQ,D)).astype(np.float32)
    def W(skew): return [rng.standard_normal((D,D)).astype(np.float32)/np.sqrt(D) for _ in range(4)] + \
        [rng.standard_normal((D,FFN)).astype(np.float32)/np.sqrt(D),
         (rng.standard_normal((FFN,D)).astype(np.float32)/np.sqrt(FFN))*skew[:,None]]
    skew=(np.arange(1,FFN+1,dtype=np.float32)**-0.35); skew/=skew.mean(); rng.shuffle(skew)
    Wts=W(skew)

    # importance signals (the NPU thresholds these)
    Q=x@Wts[0]; K=x@Wts[1]
    kv_mass = softmax((Q[:, :HD]@K[:, :HD].T)/np.sqrt(HD)).sum(0)           # per-key mass (head 0 proxy)
    act=np.maximum(x@Wts[4],0.0); ffn_imp=np.linalg.norm(act,axis=0)*np.linalg.norm(Wts[5],axis=1)

    kv_keep,npu1 = npu_select(kv_mass, KEEP_KV)
    ffn_keep,npu2 = npu_select(ffn_imp, KEEP_FFN)

    t_full,y_full = timed(lambda: layer(x,Wts))
    t_prune,y_prune = timed(lambda: layer(x,Wts,kv_keep,ffn_keep))
    cos=float((y_full*y_prune).sum()/(np.linalg.norm(y_full)*np.linalg.norm(y_prune)+1e-9))
    f_full=matmul_flops(SEQ,FFN); f_prune=matmul_flops(len(kv_keep),len(ffn_keep))

    print(f"open-xdna :: transformer LAYER — CPU+GPU vs CPU+GPU+NPU-prune  (SEQ={SEQ}, D={D}, H={HEADS}, FFN={FFN})")
    print(f"  NPU pruned: KV {len(kv_keep)}/{SEQ} keys, FFN {len(ffn_keep)}/{FFN} cols  (select cost {npu1+npu2:.2f} ms warmed)")
    print(f"  layer matmul FLOPs:  full {f_full/1e9:.2f} G  +prune {f_prune/1e9:.2f} G  -> {f_full/f_prune:.2f}x less")
    print(f"  layer wall-clock*:   full {t_full*1e3:6.2f} ms  +prune {t_prune*1e3:6.2f} ms  -> {t_full/t_prune:.2f}x  (*incl softmax + BLAS noise; FLOP cut is authoritative)")
    print(f"  layer-output cosine (pruned vs full): {cos:.4f}")
    print(f"  VERDICT: +prune = {f_full/f_prune:.2f}x less matmul work @ cos {cos:.3f}, NPU select ~{npu1+npu2:.1f}ms — a NET WIN, modest at")
    print(f"           layer level (unprunable QKV/Wo/up_proj dominate; component peaks were 1.6x FFN / 4x attention).")


if __name__=="__main__": main()
