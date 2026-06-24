# Results

All results from a fully open-source stack built/installed on the test machine — no
proprietary AMD Ryzen AI Software, no Windows.

## Test system

| | |
|---|---|
| Machine | HP Victus 16-s1xxx |
| CPU / APU | AMD Ryzen 7 8845HS (Hawk Point, Zen 4, 8c/16t) |
| NPU | XDNA1, PCI `1022:1502`, enumerated `RyzenAI-npu1`, arch `aie2`, 6×5 tiles, ~16 TOPS |
| iGPU | Radeon 780M (RADV/Vulkan) |
| RAM | 96 GB DDR5-5600 |
| OS / kernel / compiler | Ubuntu 25.10 / Linux 6.17.0-6 / GCC 15.2 |
| XRT | 2.25.0 (built from `amd/xdna-driver` source) |
| Driver | staging `amdxdna.ko` 0.15.0 (from `amd/xdna-driver`) |
| NPU firmware | `npu.sbin.1.5.5.391` |
| IRON / mlir-aie | `mlir_aie 1.3.3.dev13` |
| Peano | `llvm-aie 21.0.0` |

## NPU enumeration

```
$ xrt-smi examine
|BDF             |Name          |Architecture  |Topology  |
|[0000:06:00.1]  |RyzenAI-npu1  |aie2          |6x5       |
```

## Kernel: passthrough (`programming_examples/basic/passthrough_kernel`, 4096 bytes)

```
NPU time     (avg/min/max us): 127.5 / 120.7 / 135.9
End-to-end   (avg/min/max us): 220.6 / 209.0 / 232.2
PASS!
```

## Kernel: matrix multiply (`basic/matrix_multiplication/single_core`)

512×512×512, int16 in / int32 out, 32×32×32 tiles, verified against NumPy:

```
Avg NPU matmul time: 3956 us
Avg NPU GFLOPS:      67.8553
PASS!
```

## Heterogeneous inference benchmark — CPU / Radeon 780M / XDNA1 NPU

Goal: show that an **integrated-only** machine (CPU + iGPU + NPU, *no discrete GPU*) is a
real inference box. The discrete RTX 4070 is deliberately excluded. iGPU = AMD Radeon 780M
via Vulkan (RADV); the NVIDIA Vulkan device is excluded with `-dev Vulkan0`.

### A. Whole-model LLM inference (llama.cpp, Vulkan build, CPU vs iGPU)

Qwen2.5 Q4_K_M, 8 threads. Token generation (`tg128`) is the interactive-speed metric;
prompt processing (`pp512`) is the prefill metric.

| Model | Metric | CPU only (`-ngl 0`) | CPU + Radeon 780M | Δ (iGPU vs CPU) |
|-------|--------|---------------------|-------------------|-----------------|
| 3B coder | tg128 | 24.17 t/s | **34.54 t/s** | **+43%** |
| 3B coder | pp512 | 626 t/s* | 677 t/s | ~even |
| 7B instruct | tg128 | 11.02 t/s | **16.50 t/s** | **+50%** |
| 7B instruct | pp512 | **777 t/s** | 321 t/s | **−59%** |

\* 3B CPU pp512 had high run-to-run variance; treat as approximate.

**Finding:** the integrated Radeon 780M accelerates **token generation by 43–50%**
(memory-bandwidth bound), but on 7B **prompt prefill the CPU wins** (8× AVX-512 cores
out-GEMM the small iGPU on big batches). The optimal policy is **iGPU for decode, CPU
for prefill** — precisely the op→device split a routing layer (see "Toward device coffers")
would learn.

### B. Matmul throughput per device (the LLM primitive)

These are each device's *demonstrated* matmul throughput. **Not a controlled benchmark** —
shapes, dtypes, and frameworks differ (noted per row). Use as capability indicators, not a
single ranking.

| Device | Matmul | Throughput | How measured |
|--------|--------|-----------|--------------|
| CPU (Ryzen 7 8845HS, 8c, AVX-512) | 512³, f32, square | **155.6 GFLOPS** | NumPy / OpenBLAS |
| **XDNA1 NPU** (gen-1, ~16 TOPS) | 512³, int16→int32, square | **67.9 GFLOPS** | IRON/mlir-aie (this repo) |
| Radeon 780M (Vulkan) | LLM-shape gemv (m=4096,k=14336), f32 | ~39 GFLOPS | llama.cpp `test-backend-ops` |
| Radeon 780M (Vulkan) | LLM-shape gemv, q4_K (real weights) | ~274–517 GFLOPS | llama.cpp `test-backend-ops` |

**Finding:** the **first-gen XDNA1 NPU already does real GEMM at ~68 GFLOPS (int16)** — about
44% of an 8-core AVX-512 CPU's f32 GEMM, from a ~16-TOPS part at a fraction of the power, via
a fully open-source toolchain. This is the foundation for offloading LLM matmuls to the NPU.

### Toward device coffers (next)

The remaining tiers — **CPU + NPU** and **CPU + iGPU + NPU** — require routing LLM matmuls to
the NPU (no XDNA ggml backend exists yet; a matmul-offload bridge is in progress). The routing
policy itself reuses Elyan Labs' RAM-coffers (NUMA weight-banking) and neuromorphic
op→region ideas, retargeted from NUMA nodes to **compute units (CPU / iGPU / NPU)**.

## Reproducing

See [`docs/BRINGUP.md`](docs/BRINGUP.md). Once XRT + driver + firmware + IRON are in
place:

```bash
sudo bash scripts/run_example.sh basic/passthrough_kernel
sudo bash scripts/run_example.sh basic/matrix_multiplication/single_core
```

## Notes / honesty

- Runs currently require `sudo` (or membership in the `render` group with a fresh login)
  for `/dev/accel/accel0` access.
- The matched driver and firmware are installed live; they revert on reboot unless made
  persistent (DKMS) — see roadmap.
- These are single-kernel microbenchmarks, not end-to-end LLM throughput. Matmul is the
  dominant LLM primitive; full-model results will be added as the inference path lands.

## 3-way heterogeneous LLM benchmark (CPU + Radeon 780M + NPU, 96 GB unified)

Measured with `llama-bench` (CPU+Vulkan build, NVIDIA excluded via `-dev Vulkan0`), Qwen2.5,
pp512 / tg128, 8 threads. The NPU is the prune coprocessor (see FFN/attention prune results),
not a token-gen device.

| Model (quant) | decode CPU | decode 780M | iGPU decode win | prefill CPU | prefill 780M |
|---------------|-----------:|------------:|:---------------:|------------:|-------------:|
| Qwen2.5-3B Q4_K   | 24.8 t/s | **35.4 t/s** | **+43%** | **1685 t/s** | 693 t/s |
| Qwen2.5-7B Q4_K   | 11.6 t/s | **16.7 t/s** | **+44%** | **632 t/s**  | 302 t/s |
| Qwen2.5-14B Q3_K  |  7.2 t/s |  **9.9 t/s** | **+38%** | **408 t/s**  | 156 t/s |

**Device-specialization law (consistent across all sizes):**
- **Decode → Radeon 780M** (bandwidth-bound): +38–44% over CPU at every size.
- **Prefill → CPU** (compute-bound): 8× AVX-512 cores beat the iGPU 2.1–2.6×.
- **Prune → XDNA1 NPU** (the coprocessor): 1.6× FFN / 4× attention work-elimination at ~6.6 W.
- The optimal integrated-only engine routes **prefill→CPU, decode→780M, prune→NPU** — the
  "device coffers" policy, now backed by measurements.

**The 96 GB unified-memory win:** the 14B (6.8 GB) runs on the 780M via its 46 GB UMA — with
KV/context headroom the 8 GB discrete GPU lacks; 30B+ models run *only* on the iGPU+96 GB path.
Integrated graphics + a big RAM pool is a real large-model inference box, no discrete GPU.
