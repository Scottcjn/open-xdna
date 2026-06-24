TITLE (one line):
open-xdna: first-gen XDNA1 (Phoenix) NPU running on Linux with a 100% open stack — bring-up, hand-authored AIE kernels, and an honest map of what it's actually good at

CATEGORY: Showcase  (also fits Projects & Demos)

----- BODY -----

**TL;DR.** I brought up the **gen-1 XDNA1 (Phoenix / Hawk Point) NPU on Linux end-to-end** with a fully open-source stack — XRT + amd/xdna-driver + IRON/mlir-aie + Peano — on a Ryzen 7 8845HS (Ubuntu 25.10 / kernel 6.17). This is the generation the official LLM stacks (FastFlowLM, Lemonade, OGA Hybrid) skip in favor of XDNA2. Everything below is **measured on real silicon**, with the negatives kept in. Repo: https://github.com/Scottcjn/open-xdna

**Why.** XDNA1 got upstream kernel driver support (`amdxdna`), but the higher-level LLM toolchain was never built out for it. So the only path is hand-authoring AIE kernels via IRON/mlir-aie — which is exactly what this repo documents, so the next person on Phoenix hardware doesn't have to rediscover it.

**Bring-up (the 4 non-obvious fixes).** The repo has a full guide; the gotchas that cost the most time: (1) installing `libxrt_driver_xdna.so`, (2) using the staging `amdxdna.ko` over the in-tree one, (3) `llvm-objcopy` on PATH for the AIE ELF, (4) matching the NPU firmware (a stale `npu.sbin` → `ERT_CMD_STATE_ABORT`).

**Hand-authored AIE kernels (verified on device):**
- 512³ matmul, **~68 GFLOP/s (INT16 in / INT32 out)**, bit-exact vs NumPy. (Note: ~16 TOPS is the INT8 spec; I have no published INT16/BF16 figure to compare against, so I'm not claiming this is the definitive number — just a measured one.)
- A non-bijunctive "collapse/prune" suite (threshold + select + scalar stream-compaction). One honest finding: AIE2's `aie::shuffle` is structured-only — no runtime-indexed gather/scatter — so SIMD left-pack compaction isn't expressible and the scalar unit is the correct path (I opened a Q&A on the mlir-aie repo to confirm this with the compiler team).

**Heterogeneous inference — a measured device-specialization pattern.** Across Qwen2.5 3B/7B/14B on CPU vs the integrated Radeon 780M:
- **Decode → 780M** (+38–44%, bandwidth-bound)
- **Prefill → CPU** (2.1–2.6× faster than the iGPU, compute-bound)
- **Prune/vision → NPU** (low-power coprocessor)

For context, AMD's published Lemonade *Hybrid* mode uses **NPU-prefill → iGPU-decode**. On **XDNA1**, I measure the **CPU beating the NPU at prefill** (the NPU is dispatch-bound on GEMM). I'm *not* claiming this generalizes to XDNA2 — just reporting the XDNA1 result.

**Vision is the NPU's real home (it's a CNN engine).**
- A full conv/colorspace pipeline (rgba2gray → 3×3 filter2d → threshold → blend) runs bit-exact on the NPU.
- A SigLIP/ViT patch-embed runs bit-exact — but at base size the NPU is ~3.2× slower than CPU f32 BLAS; the win is **offload + ~6.6 W + concurrency, not raw speed**.
- **Camera-effects feasibility:** the full per-frame pipeline hits **220 FPS @1080p / 451 FPS @720p** including the host DMA round-trip. A live-webcam daemon prototype adds only ~2 ms/frame — the bottleneck is USB capture, never the NPU. Foundation for an open-source Linux "Studio-Effects"-style NPU webcam filter.

**Bigger models on integrated graphics.** gemma4:26b (17 GB — 2× an 8 GB discrete GPU's VRAM) runs at ~16.5 tok/s on the 780M via the 96 GB unified pool.

**Honest negatives (the part most benchmarks omit).** Structured layer-prune is a **net loss on gemma4** (small FFN + GQA → ~1.10× FLOP, gather overhead makes it slower). The prune wins (1.6× FFN, 4× attention) are **architecture-specific** — they need a large FFN or long-context KV. NPU pruning is not a universal speedup; measure on the target model's real dims first.

**Giving back upstream (from the bring-up):**
- amd/xdna-driver **#1447** — in-tree `amdxdna` (≤6.17) missing GET_ARRAY / AIE2 telemetry ioctls
- amd/xdna-driver **#1448** — docs PR: bring-up troubleshooting Q&A
- amd/xdna-driver **#1449** — code PR: SHIM graceful-degrade when the telemetry ioctl is unsupported
- Xilinx/mlir-aie **discussion #3218** — AIE2 runtime gather/scatter question

**Scope/honesty.** Research/community bring-up, not a production stack. XDNA1 (Phoenix) only — XDNA2 users would need porting. Numbers reproduce via the named scripts on this exact box.

Feedback very welcome — especially from anyone on Phoenix/Hawk Point hardware, and from the driver/compiler teams. 👉 https://github.com/Scottcjn/open-xdna
