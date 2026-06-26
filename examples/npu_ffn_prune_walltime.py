#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Scott Boudreaux / Elyan Labs. Commercial license: see ../COMMERCIAL.md
#
# open-xdna :: prune-then-shrink FFN — WALL-CLOCK net-win (the honest device-coffers test).
#
# npu_ffn_prune.py proved the FLOP ceiling (~1.6x, skew-gated). FLOPs are necessary but not
# sufficient: prune only wins if the saved compute beats the OVERHEAD it adds — the NPU
# threshold-collapse, the host gather of survivors, and the smaller-but-strided matmul.
# This times the full path end-to-end (median of N runs) so we know whether the FLOP saving
# is real wall-clock, not just an arithmetic ceiling.
#
# Path measured (skewed / LLM-like importance, where pruning is supposed to pay):
#   full   : act = relu(x@Wu); y = act @ Wd                    (CPU dense, the baseline)
#   prune  : importance -> NPU collapse(threshold) -> survivors -> act[:,k] @ Wd[k,:]   (CPU)
# We report wall-clock speedup AND cosine vs full, so a "win" is never accuracy-blind.
#
# HONEST framing: down_proj is the only prunable matmul here (you always pay full up_proj),
# so the wall-clock ceiling is below the whole-FFN FLOP ceiling. The NPU collapse + gather are
# fixed costs that eat into small prunes. We show where the crossover actually is.
#
# Run (root or render-group, IRON env active):
#   python3 examples/npu_ffn_prune_walltime.py

import os, shutil, time
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import In, Out, CompileTime
from aie.iron.algorithms import transform
from aie.iron.kernels._common import _make_extern, _default_source_path

SEQ, D, H = 512, 512, 2048
REPS = 7  # median of REPS timed runs per cell

_KDIR = os.path.dirname(str(_default_source_path("relu.cc")))
_DST = os.path.join(_KDIR, "collapse_rt.cc")
_SRC = os.path.join(os.path.dirname(__file__), "kernels", "collapse_rt.cc")
# Copy the kernel into the IRON source dir if absent (matches the sibling
# examples' convention; all of them ship the same examples/kernels/collapse_rt.cc).
if os.path.exists(_SRC) and not os.path.exists(_DST):
    shutil.copy(_SRC, _DST)


@iron.jit
def npu_collapse_rt(A: In, C: Out, TAU: In, *, in_size: CompileTime[int] = H):
    tile_size = in_size // 4
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]
    k = _make_extern("bf16_collapse_rt", _DST, [tile_ty, tile_ty, scalar_ty, np.int32])
    return transform(k, np.ndarray[(in_size,), np.dtype[bfloat16]], scalar_ty, tile_size=tile_size)


def npu_keep(importance, tau):
    """NPU threshold-collapse INCLUDING the host readback (.numpy()) -> boolean mask.

    The readback is part of the cost: the host needs the mask to gather survivors,
    so timing must include it. Returns the boolean keep-mask.
    """
    a_t = iron.tensor(importance.astype(bfloat16), dtype=bfloat16, device="npu")
    t_t = iron.tensor(np.array([int(round(tau * 256))], dtype=np.int32), dtype=np.int32, device="npu")
    c_t = iron.zeros_like(a_t)
    npu_collapse_rt(a_t, c_t, t_t, in_size=H)
    return c_t.numpy().astype(np.float32) != 0.0


def median_ms(fn):
    ts = []
    fn()  # warm
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def main():
    print(f"open-xdna :: prune-then-shrink FFN WALL-CLOCK (SEQ={SEQ}, D={D}, H={H}, median of {REPS})")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((SEQ, D)).astype(np.float32)
    Wu = rng.standard_normal((D, H)).astype(np.float32) / np.sqrt(D)
    # skewed / realistic importance (power-law) — the regime pruning is meant for
    skew = (np.arange(1, H + 1, dtype=np.float32) ** -0.9); skew = skew / skew.mean()
    np.random.default_rng(1).shuffle(skew)
    Wd = (rng.standard_normal((H, D)).astype(np.float32) / np.sqrt(H)) * skew[:, None]

    act = np.maximum(x @ Wu, 0.0)
    y_full = act @ Wd
    importance = np.linalg.norm(act, axis=0) * np.linalg.norm(Wd, axis=1)

    # baseline: dense down_proj wall-clock (the part prune competes against)
    t_down_full = median_ms(lambda: act @ Wd)
    t_full_ffn = median_ms(lambda: np.maximum(x @ Wu, 0.0) @ Wd)
    t_up = t_full_ffn - t_down_full  # unprunable up_proj + relu share

    print(f"\n  baseline: dense down_proj {t_down_full:.3f} ms | whole dense FFN {t_full_ffn:.3f} ms")
    print("  prune path timed = NPU collapse + readback + np.where + act/Wd gather + shrunk matmul")
    print(f"  (importance norms computed outside the timed region — amortizable/approximable)\n")
    print(f"  {'prune':>6} {'kept':>5} {'prune path':>11} {'down speedup':>13} {'FFN speedup':>12} {'cos':>7}")
    for p in [25, 50, 75, 90]:
        tau = float(np.percentile(importance, p))

        def prune_path():
            mask = npu_keep(importance, tau)                 # NPU collapse + host readback
            kept = np.where(mask)[0]                         # survivor indices
            actk = np.ascontiguousarray(act[:, kept])        # gather dynamic activations
            Wdk = np.ascontiguousarray(Wd[kept, :])          # gather survivor weight rows
            return actk @ Wdk                                # shrunk down_proj

        kept = np.where(npu_keep(importance, tau))[0]
        if len(kept) == 0:
            print(f"  {p:>4}%   (0 survivors at this threshold — skipped)")
            continue
        t_prune = median_ms(prune_path)
        kn = len(kept)
        y_p = np.ascontiguousarray(act[:, kept]) @ np.ascontiguousarray(Wd[kept, :])
        cos = float((y_full * y_p).sum() / (np.linalg.norm(y_full) * np.linalg.norm(y_p) + 1e-9))
        down_speedup = t_down_full / t_prune                 # full prune path vs dense down_proj
        ffn_speedup = t_full_ffn / (t_up + t_prune)          # incl. unprunable up_proj
        print(f"  {100*(1-kn/H):>5.0f}% {kn:>5d} {t_prune:>9.3f}ms "
              f"{down_speedup:>12.2f}x {ffn_speedup:>11.2f}x {cos:>7.4f}")

    print("\n  'prune path' is the FULL dynamic cost (collapse+readback+where+gather+matmul)")
    print("  vs the dense down_proj. A real win needs that whole path to beat dense AND hold")
    print("  cosine. Low-prune ratios lose: the strided gather breaks BLAS contiguity.")


if __name__ == "__main__":
    main()
