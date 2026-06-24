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

### P0 RESULTS (2026-06-23) — collapse primitive PROVEN on the NPU
- **ReLU collapse (keep-strong / prune-weak) PASS on XDNA1** via `ml/eltwise_unary -o relu`
  (bf16, 65536 elems, `@iron.jit` + `transform_parallel`). This is the elementwise
  non-bijunctive collapse primitive (`pse-vcipher-collapse`) executing on gen-1 silicon.
- IRON ships the rest of the top-k path as prebuilt NPU kernels: `reduce_max`/`compute_max`
  (peak for the cutoff) + `threshold` (keep top fraction, drop rest) + `softmax` (attention
  scores). Repro: `cd programming_examples/ml/eltwise_unary && python3 eltwise_unary.py -o relu`
  (run as root or render-group, with PATH incl /usr/lib/llvm-20/bin for llvm-objcopy).
- **Remaining (the net-win proof):** compose `reduce_max → threshold` into a top-k collapse,
  then measure NPU-prune (6.6W) + smaller-iGPU-matmul vs full-iGPU-matmul on wall-clock+joules.

### TOP-K NET-WIN ANALYSIS (2026-06-23) — ~3× win, validates the prune thesis

**ADVERSARIAL REVIEW NOTE (2026-06-23):** See separate analysis. Headline 3.1-3.6x is a stitched projection, not measured. Multiple structural issues likely prevent it materializing at that magnitude.

From measured components (NPU collapse 0.27ms end-to-end, iGPU 400 GFLOPS q4_K, marginal
power NPU 6.6W / iGPU 25.9W), keeping top-25% (prune 75%), gather est. 3ms (NOT measured):

| FFN matmul | Full iGPU | NPU-prune + small iGPU | Win |
|------------|-----------|------------------------|-----|
| 512×4096×4096 | 42.9 ms / 1.11 J | 14.0 ms / 0.36 J | **3.1× faster, 3.1× less energy** |
| 512×4096×11008 | 115.4 ms / 2.99 J | 32.1 ms / 0.83 J | **3.6× faster, 3.6× less energy** |

**The win is 100% from work ELIMINATED, not NPU speed** — collapse (0.27ms) is ~30–100×
cheaper than the matmul it shrinks. The device that is *weakest at compute* (M0/M0.5) becomes
a ~3× net win doing the cheap selection that makes the iGPU's matmul 4× smaller. This is the
`pse-vcipher-collapse` thesis, validated on this heterogeneous setup — and it only works
*because* the NPU isn't asked to do the heavy compute.

**Honest gates (analytical proof from measured parts, not a shipped kernel):**
1. ACCURACY unverified — assumes top-25% preserves model quality (the pse-vcipher claim);
   perf win is real, quality needs a perplexity test.
2. GATHER cost estimated (3ms), not measured — a fused impl must measure survivor compaction.
3. Collapse measured as a generic elementwise proxy; the `reduce_max→threshold` fusion is
   built-from-proven-kernels but not yet composed into one design.

### ⚠️ CORRECTION (Grok adversarial review, 2026-06-24) — the 3× was overstated

A second-model review found a real conceptual error in the analysis above. **Keep the thesis,
drop the 3× number:**
- **You can't prune what you haven't computed.** A real FFN pays FULL cost for the up/gate
  projection that *produces* the intermediate, then prunes before the down projection: cost is
  `full + 0.25·full = 1.25`, i.e. **~1.3–1.6× less FFN work — not 4×.** Attention stays full →
  whole-model speedup is far below 3×. The table above priced one matmul in isolation (wrong).
- **q4_K gather isn't free:** selecting 25% of K breaks the contiguous quant blocks that yield
  ~400 GFLOPS; dequant-on-fly loses that throughput, repack costs gather+scale+rewrite. The
  pruned matmul will run *below* 400 GFLOPS, so 10.7 ms was the most optimistic case.
- **NPU collapse may be a net latency ADDER:** the select is a tiny metadata decision the iGPU
  (which already owns the activation) can do in µs; routing it to the NPU adds dispatch + a
  cross-engine barrier.
- **Power figures were extrapolated from dense matmul**, not a tiny select; energy win likely <2×.
- **Accuracy gate dominates:** 75% global (same-mask-all-rows) pruning is aggressive structured
  pruning; per-token selection is kinder but destroys the single-dense-small-matmul assumption.

Honest restatement: the **core thesis holds** (use the low-power weak device to *eliminate*
work for the strong one), but the realistic FFN win is **~1.3–1.6× before overheads**, gated on
accuracy, and unproven until the full path (fused collapse + measured gather + perplexity) is
built and timed. The 3× lived only in the spreadsheet.

### Kernel progress (hand-authored AIE intrinsics, all verified on XDNA1)
- ✅ `collapse.cc` — compare+select collapse (`aie::ge`/`aie::select`), baked tau.
- ✅ `collapse_rt.cc` — runtime tau + on-NPU shift (`aie::sub`); sweep tau, prune 50→92%.
- ✅ `collapse_fused.cc` — **fused `reduce_max`→dynamic tau (=frac·peak)→collapse in ONE kernel**;
  peak found in-kernel via running `aie::max` + `aie::reduce_max`; bit-exact across frac.
- ✅ `shuffle_demo.cc` — `aie::reverse` (vec_perm/shuffle) runs on the NPU — the compaction
  building block proven.
- 🚧 Full top-k **stream compaction** (dense survivor pack) = prefix-sum + scatter, the hard
  SIMD frontier; building blocks (mask, select, shuffle) all proven, the pack algorithm is next.

### Remaining build (to ship the net-win claim)
Wire the fused collapse + compaction into a real FFN/attention path; **measure** the gather;
run a perplexity check at the chosen prune ratio; **prune the producer side / per-token**
to get real work reduction. Then the ~1.3–1.6× becomes shipped-and-measured, not analytical.

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
