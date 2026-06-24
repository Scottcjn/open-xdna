# open-xdna benchmark log — by model & test

Every measurement on the test box (Ryzen 7 8845HS · XDNA1 NPU · Radeon 780M · 96 GB · Ubuntu
25.10 / kernel 6.17). **Measured = real run on this hardware. Projected/Unmeasured = flagged.**
Reproduce via the `examples/` script named in each row.

---

## A. NPU hardware primitives (`examples/kernels/*.cc`, hand-authored AIE)

| Test | Result | Script |
|------|--------|--------|
| 512³ int16 matmul | ✅ ~68 GFLOP/s, bit-exact | `single_core` (mlir-aie) |
| Non-bijunctive collapse (ge+select) | ✅ prune 50→92%, bit-exact | `npu_collapse*.py` |
| Fused reduce_max→dynamic-τ→collapse | ✅ one kernel, bit-exact | `npu_collapse_fused.py` |
| Full pipeline (reduce+threshold+compact) | ✅ one kernel | `npu_pse_collapse.py` |
| Stream compaction (dense pack) | ✅ scalar-unit (SIMD left-pack = ISA wall) | `npu_compact.py` |
| Cross-tile reduce (N>4096) | ✅ N=16384 | `npu_collapse_xtile.py` |
| vec_perm / `aie::reverse` | ✅ | `npu_shuffle_demo.py` |
| Power | NPU matmul +6.6 W vs iGPU +25.9 W | `scratchpad/m05_power.py` |

## B. Qwen2.5 (3B / 7B / 14B) — 3-way device benchmark *(measured)*

| Model | decode CPU→780M | prefill CPU vs 780M |
|-------|-----------------|---------------------|
| 3B Q4_K | 24.8→**35.4 t/s (+43%)** | **CPU 1685** > 780M 693 |
| 7B Q4_K | 11.6→**16.7 t/s (+44%)** | **CPU 632** > 780M 302 |
| 14B Q3_K | 7.2→**9.9 t/s (+38%)** | **CPU 408** > 780M 156 |

**Law:** decode→780M (bandwidth), prefill→CPU (compute), prune→NPU. 14B (6.8 GB) runs on the
780M via 46 GB UMA; 30B+ iGPU-only. Script: `llama-bench` (build-vulkan), `RESULTS.md`.

## C. Synthetic transformer prune *(measured — architecture where prune wins)*

| Test | Result | Script |
|------|--------|--------|
| FFN col-prune (skewed weights) | 1.33× @ cos 0.999 (50%), 1.60× @ 0.998 (75%) | `npu_ffn_prune.py` |
| Attention KV-prune (skewed keys) | 2.0× @ cos 0.993 (50%), 3.97× @ 0.976 (75%) | `npu_attention_prune.py` |
| Full layer (d512, ff2048>d) | 1.27× matmul @ cos 0.958 | `npu_layer_prune.py` |
| Uniform/flat weights | brutal (cos 0.75 @ 50%) — needs importance skew | both above |

**Prune wins require importance skew + a large prunable fraction (big FFN or long-ctx KV).**

## D. gemma4:26b (multimodal, sees/hears) — *mixed; honest*

| Test | Result | Status |
|------|--------|--------|
| Run on the box | **16.5 tok/s decode** (ollama, GPU/780M, 17 GB in 96 GB) | ✅ measured — bigger-model-on-96GB win |
| Stock llama.cpp load | ❌ `expected 1014 tensors, got 658` (ollama repackages) | finding — needs HF-format GGUF |
| **Layer-prune** (real dims: d2816, **ff2112<d**, GQA-8) | **1.10× FLOP, 0.71× wall (SLOWER) @ cos 0.957** | ❌ NET LOSS — small FFN+GQA → little prunable |
| **KV-prune @ long ctx** (decode) | cos **1.0000** (accuracy-safe) | ⚠️ speed UNMEASURED — numpy can't show bandwidth; needs in-engine KV sparsity |

**Verdict for gemma4:** structured layer-prune does NOT help (architecture). NPU's value here =
**vision front-end + running the 17 GB model on 96 GB**. KV-prune is accuracy-safe at long
context and *should* help decode bandwidth, but that's unmeasured (needs the in-graph hook).
Scripts: `npu_gemma4_layer_prune.py`, `npu_gemma4_kv_decode.py`, `docs/MULTIMODAL_3WAY.md`.

## E. Vision (the NPU's native CNN strength) — *measured*

| Test | Result | Script |
|------|--------|--------|
| edge_detect (rgba2gray→3×3 conv→threshold) | ✅ PASS, bit-exact vs OpenCV | `vision/edge_detect` |
| SigLIP/ViT patch-embed (matmul) | ✅ bit-exact; NPU 6.6 ms vs CPU 2.0 ms (3.2× slower at base) | `npu_patch_embed.py` |
| Patch-embed scaled (M 256→8192) | NPU amortizes to ~64 GF/s, plateaus <CPU 300 GF/s | `npu_patch_embed_scaled.py` |

**Verdict:** NPU runs vision convs (its design purpose) but isn't *faster* than CPU/iGPU at base
size — the win is **offload + ~6.6 W + concurrency** (vision on NPU while CPU/iGPU run the LLM).

---

## The one-line takeaway per model

- **Qwen / general LLM:** decode→780M, prefill→CPU, big models via 96 GB. ✅
- **Synthetic / large-FFN or long-ctx model:** NPU prune is a real 1.6–4× win. ✅
- **gemma4 (small-FFN + GQA):** prune is a net loss; NPU = vision + 96 GB big-model host. ⚠️
- **Any multimodal model:** vision front-end → NPU (native CNN, offload + low power). ✅

**Honest meta-finding:** NPU pruning is **not** a universal boost — it pays only where the
prunable fraction is large (big FFN, long-context KV) *and* importance is skewed. Measure on the
target model's real dims before claiming a win. We did; the log above shows where it works and
where it doesn't.
