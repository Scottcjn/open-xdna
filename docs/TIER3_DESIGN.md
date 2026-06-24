# Tier-3 Design Proposal: NPU matmul-offload into LLM inference

Status: PROPOSAL (for adversarial review). Builds on the verified Tier-1/2 results in
[`RESULTS.md`](../RESULTS.md) and the bring-up in [`docs/BRINGUP.md`](BRINGUP.md).

## Problem

We have a working XDNA1 NPU matmul (512³ int16, ~68 GFLOPS, via IRON `@iron.jit`) but **no
XDNA ggml backend**, so no model runs on the NPU. Goal: get *real* LLM matmuls executing on
the gen-1 NPU and prove a net win in a heterogeneous CPU + Radeon-780M(Vulkan) + NPU scheme
over 96 GB unified DDR5 — *without* the discrete GPU.

## The hard truth we must design around (adversarial framing)

XDNA1 is **~16 TOPS**. Measured matmul throughput today:

| Device | f32 GEMM | q4_K (real LLM weights) |
|--------|----------|--------------------------|
| CPU (8c AVX-512) | 155 GFLOPS | n/a |
| Radeon 780M (Vulkan) | ~39 GFLOPS | ~274–517 GFLOPS |
| XDNA1 NPU | 68 GFLOPS (int16, fixed 512³) | **0 — no quant path yet** |

So the NPU only beats the iGPU on *f32* and only at a shape it can't dynamically take. For
**quantized decode (the common case) the 780M already wins decisively.** Any NPU-in-the-loop
plan must justify itself against "just use CPU+iGPU."

## Reframe: the win is perf-per-watt + concurrency, NOT peak speed

Adversarial conclusion from our own data: on raw throughput the NPU is the *weakest* of the
three (68 < 155 CPU; iGPU q4_K 274–517). Selling Tier-3 as "faster decode" loses. The two
defensible wins for a ~16-TOPS laptop NPU are:
- **perf-per-watt** — sustained matmul at a fraction of CPU/iGPU wattage → cooler, longer
  battery, frees CPU/iGPU for other work;
- **pipeline concurrency** — NPU runs prefill-FFN *while* the iGPU runs attention → wall-clock
  overlap a single device can't achieve.
Design and measure against THOSE, not against peak t/s.

## M0 GATE (do this BEFORE any ggml hook)

Offline microbenchmark: host → NPU int8 GEMM → host **round-trip including quantization and
buffer copy**, at real FFN shapes (e.g. m=seq, k=4096, n=14336), vs CPU and iGPU doing the
identical GEMM. **Kill the bridge if** the NPU round-trip can't win on wall-clock *or* on
joules-per-GEMM in isolation — marshalling/quant overhead swamping a 16-TOPS GEMM is the most
likely failure mode and must be disproven first. Persistent pre-compiled kernel only (no
per-op `@iron.jit`).

## Proposed first milestone (smallest thing that proves the thesis)

**M1 — NPU does prefill FFN GEMMs for one model, measured end-to-end vs CPU+iGPU.**
Not a full backend. A surgical offload of the largest, most NPU-favorable matmuls.

Rationale for *prefill, FFN, int8/int16*:
- **Prefill is compute-bound, batched, large-M** → matches the NPU's strength (the iGPU
  *loses* to CPU on 7B prefill per RESULTS.md, so there's actually a gap to fill here).
- **Decode is memory-bound gemv** → leave on the 780M (it wins; RESULTS.md +43–50%).
- **FFN up/down projections** are the biggest dense GEMMs and the most regular shapes.
- **int8/int16 activations+weights**: the NPU's native strength; accept a quantization step
  rather than fighting Q4_K dequant on the NPU.

## Architecture

```
                 96 GB unified DDR5 (zero-copy where possible)
   ┌──────────────┬───────────────────────┬───────────────────────┐
   │     CPU      │   Radeon 780M (Vulkan) │      XDNA1 NPU         │
   │  scheduling, │   decode gemv (q4_K),  │  prefill FFN GEMMs     │
   │  glue, quant │   attention            │  (int8/int16, large-M) │
   └──────────────┴───────────────────────┴───────────────────────┘
                         router = "device coffers"
            (RAM-coffers idea: NUMA-node banking → compute-unit banking;
             neuromorphic op→region → op→device dispatch policy)
```

### Bridge mechanism (modeled on POWER8→C4130 GPU-offload v3)
1. A thin ggml hook intercepts selected `GGML_OP_MUL_MAT` ops by name/shape (the FFN
   projections of prefill only).
2. Host quantizes the activation tile to int8/int16, marshals A (weights, pre-quantized at
   load) + B into a buffer the NPU SHIM owns.
3. NPU runs a **persistent, pre-compiled** tiled int8/int16 GEMM (NOT `@iron.jit` per call —
   compile once at load, reuse the xclbin), returns int32 → host dequant/scale.
4. Everything else stays on CPU/iGPU.

### Key decisions to defend
- **Pre-compiled persistent kernel** over `@iron.jit` per-op: JIT compile (~tens of seconds
  first call) is fatal per-token; compile a parameterized tiled GEMM once, dispatch many.
- **int8, not Q4_K, on the NPU**: dequant-on-NPU is a research project; int8 GEMM is the
  NPU's home turf. Quantize host-side once at load.
- **Offload only where the NPU is a NET WIN**: prefill FFN. If M1 shows it isn't, we say so
  and the honest result is "XDNA1 augments, it doesn't replace CPU+iGPU."

## Kill criteria (we stop if)
- M1 prefill-with-NPU is not faster than CPU-prefill + iGPU-decode wall-clock, OR
- quantization error degrades output quality beyond a fixed threshold, OR
- per-op marshalling overhead (host↔NPU copy + quant) exceeds the GEMM time saved.

## M0 RESULTS (measured 2026-06-23) — verdict: NPU-for-speed is dead; NPU-for-watts is the play

- Stock IRON `single_core` matmul **cannot compile at FFN shapes**: `32³` tiles exceed the
  AIE DMA block-descriptor range at K=4096; `64³` tiles exceed the AIE tile's 64 KB local
  memory. The demo kernel is monolithic with no host-side K-blocking → must be reengineered
  (custom host-blocked GEMM dispatching the NPU's native ~512³ int tile, accumulate on host).
- **Wall-clock floor (one 512×4096×4096 FFN block, 17.2 GFLOP):**

  | Device | time/GEMM | notes |
  |--------|-----------|-------|
  | iGPU 780M (q4_K, ~400 GFLOPS) | **43 ms** | the bar |
  | CPU (f32, 155 GFLOPS) | 111 ms | |
  | XDNA1 NPU (int16, 68 GFLOPS *compute floor*) | **253 ms** | ignores dispatch overhead → only worse |

  → On throughput the NPU is ~6× slower than the iGPU. **Do not build Tier-3 for speed.**
- **Energy estimate (the actual win):** NPU ~253 ms × ~2–3 W ≈ 0.5–0.75 J vs iGPU 43 ms ×
  ~20 W ≈ 0.86 J → NPU plausibly wins **joules-per-GEMM**, and runs concurrently while the
  iGPU does attention. **This is the only honest justification and it is unconfirmed until
  measured with power sensors.**

## Revised plan (post-M0)

1. **M0.5 — power instrumentation:** read amdgpu/RAPL/SoC power during an NPU GEMM vs iGPU
   GEMM at matched FLOPs. Confirm or kill the joules-per-GEMM win. THIS is now the gate.
2. **M1 (only if M0.5 wins) — reengineered host-blocked NPU GEMM:** custom kernel that tiles
   FFN matmuls into NPU-native int8/int16 blocks (work *with* the silicon, AltiVec-style),
   used for sustained/battery/concurrent inference — NOT as a speed path.
3. Position the NPU as a **low-power matmul coprocessor that frees the iGPU**, scoped to our
   own pipeline (custom, in-scope), not a general ggml backend.

## M0.5 RESULTS (power, measured 2026-06-23, 8845HS on AC, RAPL pkg + amdgpu)

| Window | pkg power | marginal | J/GFLOP |
|--------|-----------|----------|---------|
| idle | 7.6 W | — | — |
| NPU matmul (512³, looped) | 14.3 W | **+6.6 W** | 0.097 (6.6/68, generous) |
| iGPU matmul (q4_K) | 33.5 W | **+25.9 W** | **0.065** (25.9/400) |

**Verdict: dense GEMM on the NPU loses on energy too.** The NPU draws 4× less power but does
~6× less work → the iGPU is *more* energy-efficient per GFLOP for dense matmul. Combined with
M0 (6× slower wall-clock), **"NPU as a dense matmul offload" is dead on both axes.**

**What M0.5 *confirms* as the NPU's real value:**
- **~6.6 W absolute floor** — an always-on / background / battery niche the 26 W iGPU can't serve.
- The pruning play wins not via NPU compute-efficiency but via **work eliminated**: prune 75% at
  6.6 W → iGPU runs a 4× smaller matmul → net save = the iGPU work that never happened.

## PIVOT (post-M0): NPU as a sparse/selective coprocessor, not a GEMM engine

M0 proved the AIE array is bad at dense regular GEMM — which is precisely the workload that
*doesn't* fit 20 tiles of {VLIW vector + scalar + 64KB SRAM + flexible stream interconnect}.
That topology is built for **sparse, local, data-dependent** work. So aim the NPU at the
work that is a BAD fit for GPU/CPU and a GOOD fit for AIE, ranked by leverage × certainty:

1. **Non-bijunctive pruning / top-k collapse (HIGHEST leverage, testable now).** The NPU runs
   the `pse-vcipher-collapse` pre-filter: score K·V pairs, keep top ~25%, drop the rest
   BEFORE the dense matmul. This flips the M0 math — the NPU need not be *fast at* GEMM; it
   makes the GEMM **smaller**, so the iGPU then does a ~4× smaller dense op. A slow pre-filter
   still nets a win because it changes asymptotics, not the constant. Directly reuses
   existing AGPL IP (`pse-vcipher-collapse`).
2. **Hebbian assist (plausible, 2nd).** Local Δw = η·pre·post co-activation updates are local
   + parallel (no global backprop) → maps to independent per-tile compute. Fast-weights /
   in-context memory.
3. **GOFAI / symbolic (frontier bet).** Per-tile scalar cores + routing host production
   rules / tetranary logic / small tree-search — the branchy integer work GPUs hate. Ties to
   the symbolic-neural-bridge. Most novel, least certain (AIE is dataflow, not a CPU farm).

### Revised first experiment (replaces dense-FFN M1)
**P0 — NPU top-k/prune kernel:** implement a selective top-k collapse on the NPU via IRON,
measure (a) does it run on AIE, (b) prune ratio achievable, (c) does *NPU-prune + smaller
iGPU-matmul* beat *full iGPU-matmul* on wall-clock AND joules. Unlike dense-FFN, this CAN
win, because the win comes from work eliminated, not work accelerated.

## Open questions for review
1. Is prefill-FFN the right first target, or is there a better NPU-favorable op?
2. Can XRT give us a low-overhead persistent-kernel dispatch, or is per-call setup the wall?
3. Is the honest answer that NPU-in-the-loop is *not* worth it on XDNA1, and we should
   instead ship the clean CPU+iGPU heterogeneous result + the NPU as a standalone GEMM
   coprocessor demo?
