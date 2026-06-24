#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Scott Boudreaux / Elyan Labs. Commercial license: see ../COMMERCIAL.md
#
# open-xdna :: gemma4-dimensioned transformer layer — CPU+GPU vs CPU+GPU + NPU-prune, measured.
# Uses gemma4:26b's REAL config (d_model 2816, FFN 2112, 16 heads, GQA kv≈8) on one block.
# The full-model run is blocked (ollama's gemma4 GGUF won't load in stock llama.cpp: 1014 vs 658
# tensors), so this measures the prune on a true gemma4-shaped layer: NPU evicts low-mass KV keys
# + low-importance FFN cols; attention-inner + down_proj run dense on survivors.
# HONEST: layer-level prune (full up/gate/QKV unprunable); decode in the real model is bandwidth-
# bound so the live win differs — needs the in-graph NPU ggml hook for an exact gemma4 number.

import os, shutil, sys, time
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import In, Out, CompileTime
from aie.iron.algorithms import transform
from aie.iron.kernels._common import _make_extern, _default_source_path

# gemma4:26b real config
SEQ, D, HEADS, KV_HEADS, FFN = 512, 2816, 16, 8, 2112
HD = D // HEADS                       # 176
KEEP_KV, KEEP_FFN = 0.50, 0.50

_KDIR = os.path.dirname(str(_default_source_path("relu.cc")))
_DST = os.path.join(_KDIR, "collapse_rt.cc")
_SRC = os.path.join(os.path.dirname(__file__), "kernels", "collapse_rt.cc")
if os.path.exists(_SRC) and not os.path.exists(_DST): shutil.copy(_SRC, _DST)


def _mk_collapse(n):
    @iron.jit
    def f(A: In, C: Out, T: In, *, in_size: CompileTime[int] = n):
        ts = in_size // 4
        tt = np.ndarray[(ts,), np.dtype[bfloat16]]; st = np.ndarray[(1,), np.dtype[np.int32]]
        k = _make_extern("bf16_collapse_rt", _DST, [tt, tt, st, np.int32])
        return transform(k, np.ndarray[(in_size,), np.dtype[bfloat16]], st, tile_size=ts)
    return f

_coll = {}
def npu_keep(imp, frac):
    n = len(imp); f = _coll.setdefault(n, _mk_collapse(n))
    tau = float(np.sort(imp)[::-1][max(1, round(n*frac))-1])
    a = iron.tensor(imp.astype(bfloat16), dtype=bfloat16, device="npu")
    t = iron.tensor(np.array([int(round(tau*256))], dtype=np.int32), dtype=np.int32, device="npu")
    c = iron.zeros_like(a)
    f(a, c, t, in_size=n); f(a, c, t, in_size=n)
    t0=time.perf_counter(); f(a,c,t,in_size=n); ms=(time.perf_counter()-t0)*1e3
    return np.where(c.numpy().astype(np.float32) != 0.0)[0], ms

def softmax(z): z=z-z.max(-1,keepdims=True); e=np.exp(z); return e/e.sum(-1,keepdims=True)

def layer(x, W, kv=None, ff=None):
    Wq,Wk,Wv,Wo,Wu,Wd = W
    Q,K,V = x@Wq, x@Wk, x@Wv; ctx=np.zeros((SEQ,D),np.float32)
    for h in range(HEADS):
        kvh = (h % KV_HEADS)            # GQA: heads share kv groups
        qs=slice(h*HD,(h+1)*HD); ks=slice(kvh*HD,(kvh+1)*HD)
        Qh=Q[:,qs]; Kh=K[:,ks]; Vh=V[:,ks]
        if kv is not None: Kh,Vh = Kh[kv],Vh[kv]
        ctx[:,qs]=softmax((Qh@Kh.T)/np.sqrt(HD))@Vh
    x = x + ctx@Wo
    h = np.maximum(x@Wu, 0.0)
    x = x + (h[:,ff]@Wd[ff,:] if ff is not None else h@Wd)
    return x

def timed(fn,it=5):
    for _ in range(2): fn()
    t=time.perf_counter()
    for _ in range(it): r=fn()
    return (time.perf_counter()-t)/it, r

def main():
    rng=np.random.default_rng(0); x=rng.standard_normal((SEQ,D)).astype(np.float32)
    skew=(np.arange(1,FFN+1,dtype=np.float32)**-0.35); skew/=skew.mean(); rng.shuffle(skew)
    W=[rng.standard_normal((D,D)).astype(np.float32)/np.sqrt(D) for _ in range(4)]+\
      [rng.standard_normal((D,FFN)).astype(np.float32)/np.sqrt(D),
       (rng.standard_normal((FFN,D)).astype(np.float32)/np.sqrt(FFN))*skew[:,None]]
    Q=x@W[0]; K=x@W[1]; kvmass=softmax((Q[:,:HD]@K[:,:HD].T)/np.sqrt(HD)).sum(0)
    act=np.maximum(x@W[4],0.0); ffimp=np.linalg.norm(act,axis=0)*np.linalg.norm(W[5],axis=1)
    kv,n1=npu_keep(kvmass,KEEP_KV); ff,n2=npu_keep(ffimp,KEEP_FFN)
    tf,yf=timed(lambda:layer(x,W)); tp,yp=timed(lambda:layer(x,W,kv,ff))
    cos=float((yf*yp).sum()/(np.linalg.norm(yf)*np.linalg.norm(yp)+1e-9))
    qkv=3*2*SEQ*D*D; attn=HEADS*2*2*SEQ; o=2*SEQ*D*D; up=2*SEQ*D*FFN
    ffull=qkv+HEADS*2*2*SEQ*SEQ*HD+o+up+2*SEQ*FFN*D
    fpr =qkv+HEADS*2*2*SEQ*len(kv)*HD+o+up+2*SEQ*len(ff)*D
    print(f"open-xdna :: gemma4-shaped layer (d={D}, ffn={FFN}, {HEADS}h GQA{KV_HEADS}) CPU+GPU vs +NPU-prune")
    print(f"  NPU pruned: KV {len(kv)}/{SEQ} keys, FFN {len(ff)}/{FFN} cols  (select {n1+n2:.2f} ms warmed)")
    print(f"  layer matmul FLOPs:  full {ffull/1e9:.2f}G  +prune {fpr/1e9:.2f}G  -> {ffull/fpr:.2f}x less")
    print(f"  layer wall-clock*:   full {tf*1e3:6.1f} ms  +prune {tp*1e3:6.1f} ms  -> {tf/tp:.2f}x  (*incl softmax/BLAS)")
    print(f"  layer-output cosine: {cos:.4f}")
    print(f"  VERDICT: gemma4-shaped layer +NPU-prune = {ffull/fpr:.2f}x less matmul @ cos {cos:.3f}, NPU select ~{n1+n2:.1f}ms.")

if __name__=="__main__": main()
