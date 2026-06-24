# XDNA1 NPU bring-up on Linux — full recipe

This is the end-to-end path used to run kernels on an AMD XDNA1 (Phoenix / Hawk Point)
NPU on Ubuntu 25.10 / kernel 6.17 / GCC 15. Adapt versions to your system.

## 0. Confirm the hardware

```bash
lspci -d 1022:1502            # "AMD IPU Device" => XDNA1 present
ls /dev/accel/accel0          # accel node (created by the in-kernel amdxdna driver)
lsmod | grep amdxdna
```

If `/dev/accel/accel0` exists and `amdxdna` is loaded, the kernel side is ready. The
in-kernel driver is enough to *enumerate*; the steps below add the userspace runtime,
a matched driver for command submission, matching firmware, and the IRON compiler.

## 1. Build XRT + the XDNA SHIM + driver from source

```bash
git clone --recurse-submodules https://github.com/amd/xdna-driver.git
cd xdna-driver
# if a nested submodule (ELFIO/cxxopts) fails, re-run from inside the aiebu dir:
( cd xrt/src/runtime_src/core/common/aiebu && git submodule update --init --recursive )

sudo ./tools/amdxdna_deps.sh                 # installs build deps (boost, etc.)

# XRT base — on GCC 15 / new distros, disable warnings-as-errors:
( cd xrt/build && ./build.sh -npu -opt -disable-werror )
sudo apt-get install -y ./xrt/build/Release/xrt_*-base.deb \
                        ./xrt/build/Release/xrt_*-base-dev.deb

# XDNA plugin (SHIM + drivers + test binaries):
( cd build && ./build.sh -release )
```

> The packaging step tries to download firmware and may fail offline; that's fine — the
> libraries and drivers are already built under `build/Release/`.

## 2. Fix #1 — install the device-discovery library

XRT enumerates the NPU by scanning for `libxrt_driver_*`. The SHIM (`libvxdna.so`) alone
is **not** enough — without `libxrt_driver_xdna.so`, `xrt-smi examine` says "0 devices found".

```bash
cd xdna-driver/build/Release/opt/xilinx/xrt/lib
sudo cp libvxdna.so.* /opt/xilinx/xrt/lib/
sudo cp libxrt_driver_xdna.so.* /opt/xilinx/xrt/lib/
cd /opt/xilinx/xrt/lib
sudo ln -sf libxrt_driver_xdna.so.<ver> libxrt_driver_xdna.so.2
sudo ln -sf libxrt_driver_xdna.so.<ver> libxrt_driver_xdna.so
sudo ln -sf libvxdna.so.<ver> libvxdna.so.1
sudo ln -sf libvxdna.so.<ver> libvxdna.so
sudo ldconfig
```

Check:

```bash
source /opt/xilinx/xrt/setup.sh
xrt-smi examine        # should now list RyzenAI-npu1
```

## 3. Fix #2 — load the matched (staging) amdxdna.ko

Mainline `amdxdna.ko` (≤6.17) lacks ioctls the current SHIM uses (`aie2_query_telemetry`,
`aie2_get_array`). Symptoms when mismatched: telemetry queries return "Operation not
supported", `GET_ARRAY` returns `-EINVAL`. Use the driver built from this repo's source —
its `vermagic` matches your running kernel.

```bash
sudo modprobe -r amdxdna
sudo insmod xdna-driver/build/Release/bins/driver/amdxdna.ko
# verify: cat /sys/module/amdxdna/srcversion ; xrt-smi examine
```

(Reverts to mainline on reboot. Make persistent with the plugin's DKMS package, or a
`/etc/modules-load.d` + `extramodules` setup, once stable.)

## 4. Fix #3 — put llvm-objcopy on PATH

IRON renames a symbol in the compiled AIE2 object using `objcopy`. Peano ships none;
GNU `/usr/bin/objcopy` errors with `Unable to recognise the format` on AIE2 ELF. Any
recent `llvm-objcopy` works (it is target-agnostic for ELF symbol edits):

```bash
sudo apt-get install -y llvm        # provides /usr/lib/llvm-XX/bin/llvm-objcopy
export PATH=/usr/lib/llvm-20/bin:$PATH
```

## 5. Fix #4 — match the NPU firmware

Stale firmware aborts command submission: `ERT_CMD_STATE_ABORT`, kernel log shows
`xdna_mailbox ... ret -22` and `Command bo size ... too large`. Install the version the
runtime expects (here `1.5.5.391`):

```bash
FW=/lib/firmware/amdnpu/1502_00
sudo curl -L -o $FW/npu.sbin.1.5.5.391 \
  https://gitlab.com/kernel-firmware/drm-firmware/-/raw/amd-ipu-staging/amdnpu/1502_00/npu.sbin.1.5.5.391
sudo cp -n $FW/npu.sbin.zst $FW/npu.sbin.zst.bak 2>/dev/null || true
sudo ln -sf npu.sbin.1.5.5.391 $FW/npu.sbin     # uncompressed; loader prefers exact name
sudo rm -f $FW/npu.sbin.zst                      # remove stale compressed so it doesn't win
sudo modprobe -r amdxdna && sudo insmod xdna-driver/build/Release/bins/driver/amdxdna.ko
```

## 6. Install IRON / mlir-aie + Peano and run a kernel

```bash
bash scripts/setup_iron.sh          # mlir-aie clone + venv + wheels (mlir_aie, llvm-aie)
sudo bash scripts/run_example.sh basic/passthrough_kernel
sudo bash scripts/run_example.sh basic/matrix_multiplication/single_core
```

IRON's `@iron.jit` auto-detects NPU1 vs NPU2, so stock examples target XDNA1 without edits.

## Troubleshooting quick map

| Symptom | Fix |
|---|---|
| `xrt-smi examine` → 0 devices | Fix #1 (install `libxrt_driver_xdna.so`) |
| `Permission denied` on `/dev/accel/accel0` | run as root, or join `render` group + re-login |
| `QUERY_TELEMETRY ... not supported` / `GET_ARRAY -22` | Fix #2 (staging driver) |
| `objcopy: Unable to recognise the format` | Fix #3 (`llvm-objcopy` on PATH) |
| `ERT_CMD_STATE_ABORT` / `mailbox ret -22` | Fix #4 (match firmware version) |
