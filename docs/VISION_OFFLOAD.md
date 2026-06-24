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
- ✅ **SigLIP/ViT patch-embed wired + measured** (`examples/npu_patch_embed.py`): the patch-projection
  matmul ([256×768]@[768×768], SigLIP-base/16) runs on the NPU, **bit-exact**. HONEST measurement: at
  base size the NPU (int16, 6.6 ms / 46 GFLOP/s, dispatch-bound) is **~3.2× SLOWER than fair CPU f32
  BLAS** (2.0 ms / 148 GFLOP/s). The win is **offload + ~6.6 W + scaling** (frees CPU/iGPU for the LLM,
  gap closes at larger image/batch), **not raw latency** at SigLIP-base. Don't quote int32-numpy as the
  CPU baseline — it's unoptimized (~1.5 GFLOP/s) and makes the NPU look 34× faster, which is false.
- 🚧 **Not yet done**: full multi-stage encoder (conv stem + transformer blocks) + a batched/larger-image
  regime where the NPU's offload+power actually nets ahead; measure end-to-end encoder latency & SoC power.
- The "boost" is expected to be real *because* this is the NPU's native workload — but it is a
  projection from the proven primitives until a full vision tower is wired and measured.
