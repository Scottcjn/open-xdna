# open-xdna

**Open-source bring-up for the first-generation AMD XDNA NPU (XDNA1 / Phoenix / Hawk Point) on Linux — with a verified matrix-multiply running on the silicon.**

> The popular "LLMs on the Ryzen AI NPU under Linux" stacks — **FastFlowLM** and **AMD Lemonade** — support **only XDNA2** (Ryzen AI 300/400 "Strix"). The **first-generation XDNA1 NPU** in millions of **Ryzen 7040 / 8040 ("Phoenix" / "Hawk Point")** laptops is left behind. `open-xdna` documents and scripts the full open-source path to actually run compute on that gen-1 NPU.

If you have a Ryzen 7 7840/8840-series, Ryzen 5 7640/8640-series, or 8600G and your NPU shows up as **`RyzenAI-npu1`** (PCI device `1502`), this repo is for you.

---

## Status

| Component | State |
|-----------|-------|
| NPU enumeration via open-source XRT | ✅ Working |
| Driver/runtime bring-up (XRT + XDNA SHIM, built from source) | ✅ Working |
| Compile a kernel with IRON/mlir-aie + Peano | ✅ Working |
| **Passthrough kernel on the NPU** | ✅ **PASS** |
| **512×512×512 int16 matmul on the NPU** | ✅ **PASS — ~68 GFLOPS** |
| Full LLM inference | 🚧 In progress (matmul is the core primitive) |

Verified on: **HP Victus 16**, **Ryzen 7 8845HS** (Hawk Point), NPU `RyzenAI-npu1` (arch `aie2`, 6×5 tiles), **Ubuntu 25.10**, kernel **6.17**, **GCC 15.2**. See [`RESULTS.md`](RESULTS.md).

---

## Why this is non-trivial (the four fixes)

Getting a kernel onto an XDNA1 NPU on a current Linux needs four non-obvious steps that no single guide covers. Full detail in [`docs/BRINGUP.md`](docs/BRINGUP.md); summary:

1. **Install `libxrt_driver_xdna.so`, not just `libvxdna.so`.** XRT discovers the NPU by scanning for the `libxrt_driver_*` driver-registration library. Miss it and `xrt-smi examine` reports **"0 devices found"** even though the kernel driver is bound.
2. **Use the matched (staging) `amdxdna.ko`.** The mainline kernel driver (≤6.17) lacks newer ioctls (`aie2_query_telemetry`, `aie2_get_array`) the current XRT SHIM expects. Load the driver built from [`amd/xdna-driver`](https://github.com/amd/xdna-driver) instead.
3. **Put `llvm-objcopy` on `PATH`.** IRON renames a symbol in the compiled AIE2 object with `objcopy`. Peano ships no `objcopy`, and GNU `/usr/bin/objcopy` can't parse the AIE2 ELF (`Unable to recognise the format`). Add any recent LLVM's `llvm-objcopy` (e.g. `/usr/lib/llvm-20/bin`).
4. **Match the NPU firmware version.** Stale firmware aborts command submission (`ERT_CMD_STATE_ABORT`, `xdna_mailbox ret -22`, "Command bo size too large"). Install the firmware the runtime expects (here: `npu.sbin.1.5.5.391`).

---

## Quick start

```bash
# 0. Prereqs: amd/xdna-driver built (XRT base + SHIM + staging driver), see docs/BRINGUP.md
# 1. Install IRON/mlir-aie + Peano (downloads wheels)
bash scripts/setup_iron.sh

# 2. Match driver + firmware (one-time, needs sudo; reverts on reboot unless made persistent)
sudo bash scripts/swap_driver.sh
sudo bash scripts/install_firmware.sh 1.5.5.391

# 3. Run a kernel on the NPU
sudo bash scripts/run_example.sh basic/passthrough_kernel
sudo bash scripts/run_example.sh basic/matrix_multiplication/single_core
```

Expected matmul tail:

```
Avg NPU matmul time: 3956us.
Avg NPU gflops: 67.8553
PASS!
```

---

## Roadmap

- [x] Drive XDNA1 with fully open-source XRT
- [x] Compile + run passthrough and matmul kernels on the NPU
- [ ] **Heterogeneous inference benchmark** — run a model across **CPU + Radeon 780M (iGPU, Vulkan) + XDNA1 NPU**, deliberately *not* the discrete GPU, and chart performance: CPU-only → +iGPU → +NPU.
- [ ] NPU matmul-offload into a llama.cpp-style inference path
- [ ] Device-aware tensor routing (treating CPU/iGPU/NPU as compute "coffers")
- [ ] Make driver + firmware persistent (DKMS)
- [ ] Upstream report: mainline `amdxdna.ko` missing-ioctl gap

## Hardware: does my chip have an XDNA1 NPU?

Run `lspci -d 1022:1502` — if it lists an **AMD IPU Device**, you have XDNA1. Chips include: Ryzen 7 7840HS/U, 8840HS/U, **8845HS**; Ryzen 9 7940HS, 8945HS; Ryzen 5 7640HS/U, 8640HS, 8645HS; desktop 8600G/8700G. (The desktop **8500G/8300G have no NPU** — Zen4c die.)

## Related work / Elyan Labs

Part of ongoing heterogeneous-compute research (PSE vec_perm collapse, RAM coffers / NUMA weight banking, neuromorphic device routing) by [Elyan Labs](https://elyanlabs.ai). See also [`ram-coffers`](https://github.com/Scottcjn/ram-coffers).

## License

MIT (this repo's scripts/docs). AMD's XDNA/XRT/IRON toolchains are Apache-2.0 WITH LLVM-exception and installed separately; no AMD code or firmware is redistributed here.
