//===- compact.cc -----------------------------------------------*- C++ -*-===//
// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Scott Boudreaux / Elyan Labs.  Commercial license: COMMERCIAL.md
//
// open-xdna :: full STREAM COMPACTION on XDNA1 — dense survivor pack.
//
// out = [ all x where x >= tau, in order ]  ++  [ zeros ]
//
// This is the top-k pack that makes the prune actually shrink downstream work: survivors are
// packed contiguously at the front so the next matmul runs on a dense [0:count] slice.
//
// NOTE ON METHOD: compaction is NOT elementwise — out[k] depends on the running survivor count
// (a scan + gather), so it can't tile-stream. It runs on the AIE *scalar* unit over the whole
// in-tile vector. The vectorized form (Hillis-Steele prefix-sum of the mask + aie::shuffle
// runtime gather) is the performance optimization; this is the correct baseline.
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

void compact(bfloat16 *restrict a, bfloat16 *restrict c, int32_t *tau_q,
             const int N) {
  const float tau = ((float)(*tau_q)) / 256.0f;
  event0();
  int k = 0;
  for (int i = 0; i < N; i++) {        // scan + gather (AIE scalar unit)
    if ((float)a[i] >= tau) {
      c[k] = a[i];                     // pack survivor to the dense front
      k++;
    }
  }
  for (int j = k; j < N; j++) {        // zero-pad the tail
    c[j] = (bfloat16)0.0f;
  }
  event1();
  return;
}

extern "C" {

void bf16_compact(bfloat16 *a_in, bfloat16 *c_out, int32_t *tau_q, int32_t N) {
  compact(a_in, c_out, tau_q, N);
}

} // extern "C"
