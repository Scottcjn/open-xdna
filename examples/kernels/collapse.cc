//===- collapse.cc ----------------------------------------------*- C++ -*-===//
// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Scott Boudreaux / Elyan Labs.  Commercial license: COMMERCIAL.md
//
// open-xdna :: hand-authored AIE2 non-bijunctive COLLAPSE kernel.
//
// Keep-strong / prune-weak in AIE vector intrinsics (the PowerPC AltiVec/VSX instinct,
// AIE mnemonics):  out[i] = (x[i] >= tau) ? x[i] : 0    (THRESH_TOZERO collapse).
//
//   aie::load_v   == vec_ld          aie::ge      == vec_cmpge (mask)
//   aie::select   == vec_sel         aie::store_v == vec_st
//
// vs relu.cc which is just aie::max(x,0) — this adds the compare+select that is the heart
// of the pse-vcipher non-bijunctive collapse.
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

void collapse(bfloat16 *restrict a, bfloat16 *restrict c, const int TILE_SIZE) {
  constexpr int v_factor = 32;
  const bfloat16 tau = (bfloat16)0.5f; // threshold (baked v1; runtime RTP is the next step)
  aie::vector<bfloat16, v_factor> tau_v = aie::broadcast<bfloat16, v_factor>(tau);
  aie::vector<bfloat16, v_factor> zero_v = aie::zeros<bfloat16, v_factor>();

  event0();
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(32, 32)
  for (size_t i = 0; i < TILE_SIZE; i += v_factor) {
    aie::vector<bfloat16, v_factor> x = aie::load_v<v_factor>(a + i);
    aie::mask<v_factor> keep = aie::ge(x, tau_v);          // vector compare: x >= tau
    aie::vector<bfloat16, v_factor> out = aie::select(zero_v, x, keep); // keep ? x : 0
    aie::store_v(c + i, out);
  }
  event1();
  return;
}

extern "C" {

void bf16_collapse(bfloat16 *a_in, bfloat16 *c_out) {
  collapse(a_in, c_out, 1024);
}

} // extern "C"
