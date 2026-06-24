//===- collapse_fused.cc ----------------------------------------*- C++ -*-===//
// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Scott Boudreaux / Elyan Labs.  Commercial license: COMMERCIAL.md
//
// open-xdna :: FUSED reduce_max -> dynamic-tau non-bijunctive collapse (AIE2 intrinsics).
//
// One kernel, two passes over the tile:
//   Pass 1: running aie::max across the data, then aie::reduce_max -> peak  (the data's max)
//   Pass 2: tau = frac * peak  (DYNAMIC, derived from the data, not passed in);
//           collapse: out = (x >= tau) ? x : 0   via aie::ge + aie::select
//
// frac is the only runtime scalar (Q8 int32): "keep everything within frac of the peak".
// This is the pse-vcipher non-bijunctive collapse with a data-adaptive threshold, fully on-NPU.
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

void collapse_fused(bfloat16 *restrict a, bfloat16 *restrict c, int32_t *frac_q,
                    const int N) {
  constexpr int v_factor = 32;
  using V = aie::vector<bfloat16, v_factor>;

  // --- Pass 1: peak = max over all elements -----------------------------------
  V running = aie::broadcast<bfloat16, v_factor>((bfloat16)(-1.0e30f));
  for (int i = 0; i < N; i += v_factor) {
    V x = aie::load_v<v_factor>(a + i);
    running = aie::max(running, x);            // vector running max
  }
  bfloat16 peak = aie::reduce_max(running);    // horizontal reduce -> scalar peak

  // --- dynamic threshold: tau = frac * peak -----------------------------------
  float frac = ((float)(*frac_q)) / 256.0f;
  bfloat16 tau = (bfloat16)(frac * (float)peak);
  V tau_v = aie::broadcast<bfloat16, v_factor>(tau);
  V zero_v = aie::zeros<bfloat16, v_factor>();

  // --- Pass 2: collapse against the data-adaptive tau -------------------------
  event0();
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(32, 32)
  for (int i = 0; i < N; i += v_factor) {
    V x = aie::load_v<v_factor>(a + i);
    aie::mask<v_factor> keep = aie::ge(x, tau_v);
    aie::store_v(c + i, aie::select(zero_v, x, keep));
  }
  event1();
  return;
}

extern "C" {

void bf16_collapse_fused(bfloat16 *a_in, bfloat16 *c_out, int32_t *frac_q,
                         int32_t N) {
  collapse_fused(a_in, c_out, frac_q, N);
}

} // extern "C"
