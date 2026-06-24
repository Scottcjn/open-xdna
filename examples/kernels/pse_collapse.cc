//===- pse_collapse.cc ------------------------------------------*- C++ -*-===//
// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Scott Boudreaux / Elyan Labs.  Commercial license: COMMERCIAL.md
//
// open-xdna :: the COMPLETE pse-vcipher non-bijunctive collapse, fused in ONE AIE kernel.
//
//   in  -> [ reduce_max -> tau = frac*peak -> keep x>=tau -> pack survivors dense ] -> out
//
// One kernel call turns a score vector into its dense top-(by-threshold) survivors:
//   Pass 1 (vectorized): running aie::max + aie::reduce_max -> data peak
//   Pass 2 (scalar scan+gather): pack {x : x >= frac*peak} contiguously, zero-fill the tail
//
// frac is the only runtime scalar (Q8 int32). This is reduce + threshold + compaction fused —
// the single op behind "prune the weak, keep the strong, hand the dense survivors downstream."
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

void pse_collapse(bfloat16 *restrict a, bfloat16 *restrict c, int32_t *frac_q,
                  const int N) {
  constexpr int v_factor = 32;
  using V = aie::vector<bfloat16, v_factor>;

  // --- Pass 1: peak (vectorized) ---
  V running = aie::broadcast<bfloat16, v_factor>((bfloat16)(-1.0e30f));
  for (int i = 0; i < N; i += v_factor)
    running = aie::max(running, aie::load_v<v_factor>(a + i));
  float peak = (float)aie::reduce_max(running);

  // --- dynamic threshold ---
  float tau = (((float)(*frac_q)) / 256.0f) * peak;

  // --- Pass 2: collapse + compact (scan + gather) ---
  event0();
  int k = 0;
  for (int i = 0; i < N; i++) {
    if ((float)a[i] >= tau) {
      c[k] = a[i];     // pack survivor to the dense front
      k++;
    }
  }
  for (int j = k; j < N; j++)
    c[j] = (bfloat16)0.0f;   // zero-fill tail
  event1();
  return;
}

extern "C" {

void bf16_pse_collapse(bfloat16 *a_in, bfloat16 *c_out, int32_t *frac_q,
                       int32_t N) {
  pse_collapse(a_in, c_out, frac_q, N);
}

} // extern "C"
