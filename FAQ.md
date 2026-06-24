# open-xdna FAQ — running LLMs / compute on the AMD XDNA1 NPU on Linux

High-intent questions about the first-generation AMD Ryzen AI NPU (XDNA1 / Phoenix /
Hawk Point) on Linux, answered honestly. If you found this by searching an error string,
jump to [Troubleshooting](#troubleshooting).

## Can the AMD XDNA1 NPU run LLMs on Linux?

Partially, and honestly: there is **no turnkey LLM runtime** for XDNA1 on Linux (FastFlowLM
and AMD Lemonade are **XDNA2-only** — Strix/Krackan). But the NPU itself is fully usable from
open source: `open-xdna` brings up the gen-1 NPU with AMD's open XRT + XDNA driver + IRON/
mlir-aie, and demonstrates real compute on the silicon — a verified 512³ matmul (~68 GFLOPS),
a 2-layer MLP forward pass, and a non-bijunctive prune/collapse kernel. A full model does not
yet run end-to-end on the NPU (no XDNA ggml backend exists).

## Does the Ryzen 7 8845HS (or 7840/8840) NPU work on Linux?

Yes. The Phoenix/Hawk-Point NPU enumerates (`lspci -d 1022:1502` → "AMD IPU Device"), the
in-kernel `amdxdna` driver (Linux 6.14+) creates `/dev/accel/accel0`, and with the userspace
from this repo it runs kernels. Verified on Ubuntu 25.10 / kernel 6.17 / GCC 15.

## Is the XDNA1 NPU faster than the Radeon 780M iGPU for LLM inference?

**No.** Measured here: the NPU is ~6× slower than the 780M at dense matmul and uses more
energy per GFLOP for dense work. Its real value is **low absolute power (~6.6 W vs the iGPU's
~26 W)** and doing cheap **selection/pruning** that lets the iGPU do *less* work. Use the iGPU
(Vulkan) for the matmuls; consider the NPU for low-power always-on or prune/collapse steps.

## Is this a FastFlowLM / Lemonade alternative for gen-1?

It's the open-source *bring-up and experimentation* path for the generation those tools skip.
It is not a drop-in `ollama`-style server. If you have a Phoenix/Hawk-Point chip and want the
NPU usable on Linux at all, this is the starting point.

## What can the XDNA1 NPU actually do well?

Sparse, local, data-dependent work — not big dense GEMM. Confirmed on silicon: reductions
(`reduce_max`), elementwise activations (`relu`), and non-bijunctive top-k collapse
(keep-strong / prune-weak). The AIE2 vector ISA (`aie::load_v`, `aie::mul`, `aie::add`,
`aie::max`, `aie::select`) is the same primitive class as PowerPC AltiVec / VSX, so custom
kernels are hand-authorable in C++ intrinsics and compiled with Peano.

## Which AMD chips have an XDNA1 NPU?

Ryzen 7 7840HS/U, 8840HS/U, **8845HS**; Ryzen 9 7940HS, 8945HS; Ryzen 5 7640HS/U, 8640HS,
8645HS; desktop 8600G / 8700G. The desktop **8500G / 8300G have NO NPU** (Zen4c die).

## Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| `xrt-smi examine` → **0 devices found** | XRT needs `libxrt_driver_xdna.so` on its lib path (not just `libvxdna.so`). See `docs/BRINGUP.md` fix #1. |
| `Permission denied` on `/dev/accel/accel0` | Run as root, or join the `render` group and re-login. |
| `aie2_query_telemetry`/`GET_ARRAY` **Operation not supported** / `-EINVAL` | Mainline `amdxdna.ko` (≤6.17) lacks ioctls the current SHIM needs — load the staging driver. Fix #2. |
| **`objcopy: Unable to recognise the format`** of an AIE `.o` | GNU objcopy can't parse AIE2 ELF; put `llvm-objcopy` on PATH (`/usr/lib/llvm-XX/bin`). Fix #3. |
| **`ERT_CMD_STATE_ABORT`** / `xdna_mailbox ret -22` / "Command bo too large" | NPU firmware too old — install the version the runtime expects (e.g. `npu.sbin.1.5.5.391`). Fix #4. |
| Host test link error `__cxa_call_terminate@CXXABI_1.3.15` | C++ ABI mismatch — build host with `CC=gcc-15 CXX=g++-15` (match XRT's GCC), or run the pure-JIT `.py` path. |
| `aie.dma_bd Size exceeds [1:64]` / `allocated buffers exceeded available memory` | Matmul shape too large for the demo kernel — block-tile host-side into NPU-native (~512³) tiles. |

## Where to start

`README.md` (overview) → `docs/BRINGUP.md` (full recipe) → `RESULTS.md` (measured numbers) →
`examples/` (runnable: `npu_tiny_mlp.py`, `npu_collapse.py`).
