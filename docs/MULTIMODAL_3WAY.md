# The 3-processor multimodal box (CPU + Radeon 780M + XDNA1 NPU, 96 GB)

The end goal: a multimodal model (image + audio + text) running across all three integrated
compute units over the 96 GB unified pool — no discrete GPU. Here's what's **measured** vs
what's **integration roadmap**, kept honest.

## Measured

| Capability | Unit | Result | Status |
|------------|------|--------|--------|
| **Bigger LLM on 96 GB** | 780M (via ollama) | **gemma4:26b (17 GB) @ 16.5 tok/s decode** | ✅ measured |
| Text decode | 780M | +38–44% over CPU (3B/7B/14B) | ✅ measured |
| Text prefill | CPU | 2.1–2.6× faster than 780M | ✅ measured |
| **"Sees" — vision front-end** | NPU | conv pipeline + SigLIP patch-embed, bit-exact | ✅ measured |
| Prune coprocessor | NPU | 1.6× FFN / 4× attention work-elimination | ✅ measured |

**The 96 GB win, concretely:** gemma4:26b is **17 GB — 2× the 8 GB NVIDIA's entire VRAM** — yet
runs at a usable 16.5 tok/s on the 780M because the iGPU draws from the 96 GB pool. The
integrated box runs models the discrete card can't hold; 30B+ stays iGPU-only.

## Integration roadmap (honest — not yet measured)

- **Controlled Gemma CPU-vs-780M split:** `gemma4` is a newer arch our `build-vulkan` llama.cpp
  (ggml 0.9.5) can't load — ollama runs it, but the controlled prefill-CPU/decode-780M benchmark
  needs a llama.cpp rebuild with gemma4 support.
- **Vision tower → NPU end-to-end:** patch-embed runs on the NPU (measured); wiring a full
  SigLIP/ViT conv stem + encoder blocks and measuring offloaded-encoder latency/power is next.
  (At base size the NPU is dispatch-bound and ~3.2× slower than CPU f32 BLAS — the win is
  offload + ~6.6 W + concurrency, not raw speed; see VISION_OFFLOAD.md.)
- **"Hears" — audio:** Ryzen AI NPUs do audio inference too, but this repo has **not** tested an
  audio path. Not claimed until measured.

## The honest shape of the thesis

A multimodal model on this box would run **vision front-end on the NPU** (its native CNN work,
at ~6.6 W, concurrent), **prefill on the CPU**, **decode on the 780M**, with the NPU also pruning
attention/FFN — all over 96 GB shared. The *components* are measured; the *full pipeline wiring*
(one model, three units, end-to-end, timed) is the integration build. We don't claim the
end-to-end win until it's wired and measured — but every piece it rests on is proven on silicon.


## gemma4:26b — measured, and an honest packaging finding (2026-06-24)

- ✅ **gemma4:26b runs at 16.5 tok/s decode** on the integrated box via **ollama** (GPU-accelerated
  on the 780M; 17 GB in the 96 GB pool). The bigger-multimodal-model-on-96GB result holds — a model
  2× the 8 GB discrete card's VRAM, usable on integrated graphics.
- ⚠️ **The ollama gemma4 GGUF will NOT load in stock llama.cpp** — even after rebuilding latest master
  (which *does* ship gemma4 support): `done_getting_tensors: wrong number of tensors; expected 1014,
  got 658`. Ollama repackages the model (and likely splits the vision/projector tensors into a separate
  blob), so its GGUF diverges from mainstream llama.cpp's gemma4 definition. A rebuild doesn't fix it;
  a HuggingFace-format gemma GGUF would be needed for the controlled CPU-vs-780M split.
- **Device split + prune still apply**: prefill→CPU / decode→780M is measured on Qwen (3-way sweep)
  and holds for any decoder LLM. NPU-prune is measured (attn 2–4×, FFN 1.6×, layer 1.27×). HONEST
  caveat: **decode is memory-bandwidth-bound**, so prune helps decode mainly via reduced KV-cache
  traffic (KV-prune) and skipped weight reads, less via FLOPs — it helps *prefill* most. A precise
  gemma4-with-prune throughput needs the in-graph NPU hook (the integration build), not a projection.


## gemma4 + NPU-prune — MEASURED on gemma4's real dims (honest negative result)

Ran the prune on a true gemma4-shaped layer (d=2816, FFN=2112, 16 heads, GQA-8 — gemma4:26b's
actual config), 50% KV + 50% FFN prune (`examples/npu_gemma4_layer_prune.py`):

| | full | +NPU-prune |
|---|---|---|
| matmul FLOPs | 47.6 G | 43.2 G — **only 1.10× less** |
| wall-clock (CPU BLAS) | 266 ms | 377 ms — **0.71× (SLOWER)** |
| output cosine | — | 0.957 |

**Verdict: structured layer-prune does NOT help gemma4.** Its FFN (2112) is *smaller* than
d_model (2816) and it uses GQA, so the unprunable projections (QKV/Wo/up-gate) dominate — only
~10% of the layer is prunable, and the gather overhead to densify survivors exceeds that, making
the pruned path a **net loss** (and you'd pay a 0.957 cosine cost for it).

**What this means honestly:** the earlier prune wins (4× attention, 1.6× FFN) were
**architecture-specific** — they need long context (big KV → KV-prune) or a large FFN. gemma4's
balanced small-FFN + GQA design isn't a prune target. **For gemma4, the NPU's value is the vision
front-end (multimodal, its native CNN work) + running the 17 GB model on the 96 GB iGPU — not
pruning.** Prune is a tool for the *right* architecture, not a universal win. (KV-prune may still
help gemma4 *decode* via reduced cache bandwidth at long context — a separate, bandwidth-bound
measurement, not this compute-bound layer test.)
