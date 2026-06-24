#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Scott Boudreaux / Elyan Labs. Commercial license: see ../COMMERCIAL.md
#
# open-xdna :: the COMPLETE pse-vcipher collapse in ONE NPU kernel (pse_collapse.cc).
# in -> reduce_max -> tau=frac*peak -> keep x>=tau -> dense pack -> out.  One call.

import os, shutil, sys
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import In, Out, CompileTime
from aie.iron.algorithms import transform
from aie.iron.kernels._common import _make_extern, _default_source_path

N = 4096
FRAC = 0.5
_KDIR = os.path.dirname(str(_default_source_path("relu.cc")))
_DST = os.path.join(_KDIR, "pse_collapse.cc")
_SRC = os.path.join(os.path.dirname(__file__), "kernels", "pse_collapse.cc")
if os.path.exists(_SRC) and not os.path.exists(_DST):
    shutil.copy(_SRC, _DST); print(f"installed pse_collapse.cc -> {_DST}")


@iron.jit
def npu_pse(A: In, C: Out, FRAC_: In, *, in_size: CompileTime[int] = N):
    tile_ty = np.ndarray[(in_size,), np.dtype[bfloat16]]
    scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]
    k = _make_extern("bf16_pse_collapse", _DST, [tile_ty, tile_ty, scalar_ty, np.int32])
    return transform(k, np.ndarray[(in_size,), np.dtype[bfloat16]], scalar_ty, tile_size=in_size)


def main():
    rng = np.random.default_rng(0)
    scores = rng.uniform(-3.0, 3.0, size=(N,)).astype(bfloat16)
    frac_q = int(round(FRAC * 256.0))
    a_t = iron.tensor(scores, dtype=bfloat16, device="npu")
    f_t = iron.tensor(np.array([frac_q], dtype=np.int32), dtype=np.int32, device="npu")
    c_t = iron.zeros_like(a_t)
    npu_pse(a_t, c_t, f_t, in_size=N)
    out = c_t.numpy().astype(np.float32)

    sf = scores.astype(np.float32)
    peak = float(sf.max())
    tau = (frac_q / 256.0) * peak
    survivors = sf[sf >= tau]
    ref = np.zeros(N, dtype=np.float32); ref[:len(survivors)] = survivors
    count = len(survivors); out_count = int(np.count_nonzero(out != 0))
    dense_ok = np.allclose(out[:count], ref[:count], atol=0.08)
    tail_ok = np.all(out[count:] == 0)
    print(f"open-xdna :: full pse-vcipher collapse in ONE NPU kernel (N={N}, frac={FRAC})")
    print(f"  in-kernel: peak={peak:.2f} -> tau={tau:.2f} -> packed {count}/{N} survivors (pruned {100*(1-count/N):.1f}%)")
    print(f"  first 6 dense: {out[:6]}")
    print(f"  dense-front matches: {dense_ok}   tail all-zero: {tail_ok}   count match: {out_count==count}")
    ok = dense_ok and tail_ok and out_count == count
    print("  RESULT:", "PASS — full collapse (reduce+threshold+compact) in one NPU kernel" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
