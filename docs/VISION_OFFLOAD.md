# NPU vision offload — the multimodal front-end belongs on the NPU

**Thesis:** for a multimodal model (image + text), put the **vision front-end on the XDNA1
NPU** and keep text generation on the CPU/iGPU. This is the NPU's *designed* purpose — Ryzen AI
NPUs exist for edge vision/CNN inference — so unlike text decode (memory-bound, where the NPU
loses), the vision path is **compute-bound conv work the AIE array is built for.**

## Why vision, not text

| | text decode | vision front-end |
|---|---|---|
| bottleneck | memory bandwidth | compute (convs/matmuls) |
| NPU fit | poor (M0/M0.5: NPU loses) | **strong** (AIE = CNN engine) |
| parallelism | low (gemv) | high (patches/channels) |
| concurrency | — | runs while CPU/iGPU do the LLM |
| power | — | ~6.6 W |

The NPU stops being a reluctant matmul unit and becomes what it was made for.

## Proven on silicon (this repo)

A full image-processing pipeline runs on the gen-1 NPU today — `programming_examples/vision/
edge_detect` (`rgba2gray → 3×3 Laplacian filter2d → threshold`), verified vs an OpenCV reference:

```
$ python3 edge_detect.py -W 512 -H 512
Number of differences: 1942, average L1 error: 0.96
PASS!
```

IRON ships the vision/CNN kernel set the front-end needs: `conv2dk1`, `conv2dk3` (incl
depthwise + ReLU + pooling variants), `conv2dk1_i8`, `filter2d`, `rgba2gray`, `gray2rgba`,
`rgba2hue`, plus `softmax`/`gelu`/`silu` for the encoder blocks. These are exactly the ops a
ViT/CNN image tower (patch-embed conv, conv stem, channel projections) is made of.

## The hook (proposed integration)

For a multimodal model with a vision encoder (PaliGemma / Gemma-vision / LLaVA-style towers):
1. **Image preprocessing + patch-embed conv → NPU** (`conv2dk1`/`conv2dk3`/`filter2d`).
2. **Vision-encoder early conv/projection stages → NPU**, streaming patch tiles.
3. Hand off the image embeddings to the text path: **decode → Radeon 780M, prefill → CPU.**
4. The NPU also runs the **prune/collapse coprocessor** on attention/FFN (see TIER3_DESIGN.md).

Net: a 4-way split on the integrated SoC — **NPU = vision + prune, 780M = decode, CPU =
prefill** — all over the 96 GB unified pool, no discrete GPU.

## Honest status

- ✅ **Proven**: conv / 2D-filter / colorspace vision kernels run on the NPU (edge_detect PASS).
  These are the building blocks of a vision tower.
- 🚧 **Not yet done**: wiring a *specific* model's vision encoder (e.g., a SigLIP/ViT tower) to
  these NPU kernels end-to-end, and measuring the offloaded-encoder latency/power vs CPU/iGPU.
  That's the integration build — the primitives are in place; the model-specific hook is next.
- The "boost" is expected to be real *because* this is the NPU's native workload — but it is a
  projection from the proven primitives until a full vision tower is wired and measured.
