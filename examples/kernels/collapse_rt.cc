//===- collapse_rt.cc -------------------------------------------*- C++ -*-===//
// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Scott Boudreaux / Elyan Labs.  Commercial license: COMMERCIAL.md
//
// open-xdna :: runtime-tau non-bijunctive COLLAPSE with on-NPU shift (AIE2 intrinsics).
//
//   out[i] = (x[i] >= tau) ? x[i] : 0      tau is a RUNTIME scalar (Q8 fixed-point int32).
//
// Demonstrates the on-NPU shift via aie::sub (the AltiVec vec_sub equivalent):
//   x - tau >= 0   <=>   x >= tau
// so the threshold test is computed entirely on the AIE vector unit, tau supplied at runtime.
//
//   aie::broadcast (vec_splat) · aie::sub (vec_sub) · aie::ge (vec_cmpge→mask) · aie::select (vec_sel)
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

// tau passed as Q8 fixed-point: tau = (*tau_q) / 256.0
void collapse_rt(bfloat16 *restrict a, bfloat16 *restrict c, int32_t *tau_q,
                 const int N) {
  constexpr int v_factor = 32;
  bfloat16 tau = (bfloat16)(((float)(*tau_q)) / 256.0f);
  aie::vector<bfloat16, v_factor> tau_v = aie::broadcast<bfloat16, v_factor>(tau);
  aie::vector<bfloat16, v_factor> zero_v = aie::zeros<bfloat16, v_factor>();

  event0();
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(32, 32)
  for (int i = 0; i < N; i += v_factor) {
    aie::vector<bfloat16, v_factor> x = aie::load_v<v_factor>(a + i);
    aie::vector<bfloat16, v_factor> diff = aie::sub(x, tau_v);   // ON-NPU shift (vec_sub)
    aie::mask<v_factor> keep = aie::ge(diff, zero_v);            // x - tau >= 0
    aie::vector<bfloat16, v_factor> out = aie::select(zero_v, x, keep); // keep ? x : 0
    aie::store_v(c + i, out);
  }
  event1();
  return;
}

extern "C" {

void bf16_collapse_rt(bfloat16 *a_in, bfloat16 *c_out, int32_t *tau_q, int32_t N) {
  collapse_rt(a_in, c_out, tau_q, N);
}

} // extern "C"
