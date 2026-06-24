//===- shuffle_demo.cc ------------------------------------------*- C++ -*-===//
// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Scott Boudreaux / Elyan Labs.  Commercial license: COMMERCIAL.md
//
// open-xdna :: prove the vec_perm / lane-permute primitive runs on the AIE vector unit.
// Reverses each 32-lane vector (a fixed permutation) via aie::reverse — the building block
// for top-k survivor compaction (the full data-dependent pack is prefix-sum+scatter, the
// remaining frontier; this verifies the permute engine itself works on XDNA1).
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

void shuffle_demo(bfloat16 *restrict a, bfloat16 *restrict c, const int N) {
  constexpr int v_factor = 32;
  event0();
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(32, 32)
  for (int i = 0; i < N; i += v_factor) {
    aie::vector<bfloat16, v_factor> x = aie::load_v<v_factor>(a + i);
    aie::vector<bfloat16, v_factor> r = aie::reverse(x); // vec_perm-class lane permutation
    aie::store_v(c + i, r);
  }
  event1();
  return;
}

extern "C" {

void bf16_shuffle_demo(bfloat16 *a_in, bfloat16 *c_out) {
  shuffle_demo(a_in, c_out, 1024);
}

} // extern "C"
