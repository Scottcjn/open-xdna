<div align="center">

# ⚡ open-xdna

### Running AI on the AMD NPU that AMD says you can't.

**The first-generation XDNA1 NPU — "Phoenix" / "Hawk Point" — in millions of Ryzen laptops, brought up on Linux with 100% open source. Models running on the silicon. Custom vector kernels hand-authored in the AIE ISA.**

[![BCOS Certified](https://img.shields.io/badge/BCOS-Certified-brightgreen?style=flat)](BCOS.md)
![License](https://img.shields.io/badge/license-AGPLv3%20%2B%20commercial-blue)
![NPU](https://img.shields.io/badge/NPU-XDNA1%20Phoenix%20%C2%B7%20RyzenAI--npu1-red)
![Status](https://img.shields.io/badge/gen--1%20NPU-running%20on%20Linux-brightgreen)
![Stack](https://img.shields.io/badge/100%25-open%20source-success)
![Built from](https://img.shields.io/badge/Ubuntu-25.10%20%C2%B7%20kernel%206.17%20%C2%B7%20GCC%2015-orange)

</div>

---

> **FastFlowLM** and **AMD Lemonade** — the stacks everyone points to for "LLMs on the Ryzen AI NPU" — support **XDNA2 only**. If you own a **Ryzen 7040 / 8040** laptop, AMD's answer for your NPU on Linux is *"not available — use the GPU."*
>
> **`open-xdna` is the answer that says "here's how."**

```
$ xrt-smi examine
  [0000:06:00.1]   RyzenAI-npu1   aie2   6x5      ←  the chip they left behind, alive on Linux
```

## 🏆 Proven on silicon (every row measured, on a real Ryzen 7 8845HS)

| Milestone | Result |
|-----------|:------:|
| 🔌 Gen-1 NPU driven by **fully open-source** XRT + driver + firmware | ✅ |
| ♻️ Survives reboot + kernel updates (DKMS) | ✅ |
| ⚙️ `512³` int16 matmul on the NPU | ✅ **~68 GFLOPS** |
| 🧠 A real model's forward pass (2-layer MLP) on the NPU | ✅ bit-exact |
| ✂️ Non-bijunctive **collapse** (keep-strong / prune-weak) | ✅ prune 50→92% |
| 🛠️ **Hand-authored AIE vector kernel** (`aie::ge`+`aie::select`) | ✅ first compile |
| 🔁 Fused `reduce_max`→dynamic-τ collapse in **one** kernel | ✅ data-adaptive |
| 🔀 `aie::reverse` (vec_perm / shuffle) on the NPU | ✅ |
| 🎯 **Full collapse** (reduce+threshold+compact) in **one** AIE kernel | ✅ |
| 📈 **Measured FFN net-win** (NPU prune → dense down_proj) | ✅ **1.6× @ cos 0.998** |

## 😈 Why this exists

Hardware vendors retire silicon with *software*, not screwdrivers. AMD moved on to XDNA2 and told gen-1 owners the NPU "isn't available" for LLMs on Linux. The chip is fine. The driver's in the kernel. The compiler exists. What was missing was a recipe — so here's one, end to end, with receipts.

## 🔧 The four fixes nobody documented together

Getting a kernel onto an XDNA1 NPU on a current Linux needs four non-obvious moves (full detail: [`docs/BRINGUP.md`](docs/BRINGUP.md), error-string fixes in the [**FAQ**](FAQ.md)):

1. **Install `libxrt_driver_xdna.so`** (not just `libvxdna.so`) — else `xrt-smi` reports *"0 devices found"*.
2. **Load the matched staging `amdxdna.ko`** — mainline (≤6.17) lacks ioctls the SHIM needs.
3. **Put `llvm-objcopy` on `PATH`** — GNU objcopy can't parse the AIE2 ELF.
4. **Match the NPU firmware** — stale firmware aborts commands (`ERT_CMD_STATE_ABORT`).

## 🚀 Quick start

```bash
bash scripts/setup_iron.sh                      # IRON/mlir-aie + Peano
sudo bash scripts/swap_driver.sh                # matched staging driver
sudo bash scripts/install_firmware.sh 1.5.5.391 # matching NPU firmware
sudo bash scripts/run_example.sh basic/matrix_multiplication/single_core   # 68 GFLOPS on the NPU
```

Run the hand-authored kernels:
```bash
python3 examples/npu_tiny_mlp.py            # a model's matmuls on the NPU
python3 examples/npu_collapse_fused.py      # fused reduce_max -> dynamic-tau collapse
python3 examples/npu_collapse_runtime.py    # runtime-tau collapse (on-NPU aie::sub shift)
python3 examples/npu_shuffle_demo.py        # aie::reverse (vec_perm) on the NPU
python3 examples/npu_pse_collapse.py        # FULL collapse (reduce+threshold+compact) in ONE kernel
python3 examples/npu_ffn_prune.py           # MEASURED FFN net-win + accuracy tradeoff
```

## 🧬 The flex: you can hand-write AIE kernels like AltiVec

The AIE2 vector ISA is the **same primitive class as PowerPC AltiVec/VSX** — just new mnemonics. The non-bijunctive collapse, hand-authored ([`examples/kernels/collapse.cc`](examples/kernels/collapse.cc)):

```cpp
aie::vector<bfloat16,32> x   = aie::load_v<32>(a + i);    // vec_ld
aie::mask<32>            keep = aie::ge(x, tau_v);         // vec_cmpge  → mask
aie::vector<bfloat16,32> out  = aie::select(zero_v, x, keep); // vec_sel: keep ? x : 0
aie::store_v(c + i, out);                                 // vec_st
```

Porting your own SIMD kernels? [`docs/ALTIVEC_TO_AIE.md`](docs/ALTIVEC_TO_AIE.md) maps the whole vocabulary.

## 📊 The honest part

This is **not** a turnkey LLM server, and the NPU is **not** a fast matmul engine — measured, it's ~6× slower than the integrated Radeon 780M at dense GEMM and loses on energy-per-GFLOP for dense work ([`RESULTS.md`](RESULTS.md)). Its real value is a **~6.6 W power floor** and cheap **pruning/selection** that lets a stronger device do less work. A prune-then-shrink FFN nets ~**1.3–1.6×** less work (not the naive 4× — you still pay the producer projection), gated on accuracy. We measure before we claim, and we mark every frontier.

## 🗺️ Does my chip have an XDNA1 NPU?

`lspci -d 1022:1502` → "AMD IPU Device" = yes. **7840HS/U · 8840HS/U · 8845HS · 7940HS · 8945HS · 7640HS/U · 8640HS · 8645HS · 8600G/8700G.** (Desktop **8500G/8300G have no NPU** — Zen4c die.)

## 🔗 Related work — Elyan Labs

Heterogeneous-compute research (PSE non-bijunctive collapse, RAM coffers / NUMA weight banking, neuromorphic device routing) by [**Elyan Labs**](https://elyanlabs.ai). See also [`ram-coffers`](https://github.com/Scottcjn/ram-coffers) · [`pse-vcipher-collapse`](https://github.com/Scottcjn/pse-vcipher-collapse).

## 📜 License

**AGPLv3** (this repo's original work) — see [`LICENSE`](LICENSE). A **commercial / proprietary license** (no copyleft) is available for closed-source or SaaS use: [`COMMERCIAL.md`](COMMERCIAL.md). AMD's XDNA/XRT and IRON/mlir-aie toolchains are Apache-2.0 WITH LLVM-exception and installed, not redistributed.
