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

## Open questions for review
1. Is prefill-FFN the right first target, or is there a better NPU-favorable op?
2. Can XRT give us a low-overhead persistent-kernel dispatch, or is per-call setup the wall?
3. Is the honest answer that NPU-in-the-loop is *not* worth it on XDNA1, and we should
   instead ship the clean CPU+iGPU heterogeneous result + the NPU as a standalone GEMM
   coprocessor demo?
