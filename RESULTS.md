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
