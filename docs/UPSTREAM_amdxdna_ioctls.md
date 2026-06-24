# Upstream report: mainline `amdxdna` (kernel ≤6.17) missing ioctls expected by current XRT SHIM

**Summary.** On a stock mainline kernel, the in-tree `amdxdna` driver does **not** implement
the `GET_ARRAY` and AIE2 telemetry ioctls that the current XRT XDNA SHIM (XRT 2.25, the
`amd/xdna-driver` HEAD plugin) issues. The result is that telemetry / array queries fail with
`EOPNOTSUPP`/`EINVAL`, and users must build and side-load the repo's **staging** `amdxdna.ko`
to get a fully functional NPU. This asks for either a backport to the in-tree driver or a
documented minimum-driver-version contract (and ideally graceful degradation in XRT).

## Environment

| | |
|---|---|
| Hardware | AMD Ryzen 7 8845HS — XDNA1 / Phoenix NPU, PCI `1022:1502`, `RyzenAI-npu1` |
| OS / kernel | Ubuntu 25.10, **kernel 6.17.0-6-generic** (mainline `amdxdna`) |
| In-tree driver | `/lib/modules/6.17.0-6-generic/kernel/drivers/accel/amdxdna/amdxdna.ko` |
| Userspace | XRT **2.25.0** + XDNA SHIM built from `amd/xdna-driver` HEAD (June 2026) |
| Staging driver | `amd/xdna-driver` `build/Release/bins/driver/amdxdna.ko` (DKMS) |

## Symptom (mainline in-tree driver loaded)

`xrt-smi examine` enumerates the device, and basic BO/exec paths work, but the SHIM's
telemetry / array-info queries fail. From the repo's `shim_test`:

```
ioctl(QUERY_TELEMETRY) failed: Operation not supported
DRM_IOCTL_AMDXDNA_GET_ARRAY IOCTL failed (err=-22): Invalid argument
QUERY_TELEMETRY with header-only buffer should fail EINVAL, got 95     # 95 = EOPNOTSUPP
```

`shim_test` result on mainline: **25 subtests pass**, the telemetry/get_array subtests fail.

## Root cause

The HEAD XRT SHIM calls newer driver entry points — `amdxdna_drm_get_array_ioctl` /
`aie2_get_array` and the AIE2 telemetry query (`aie2_query_telemetry` /
`amdxdna_get_telemetry`) — that the **in-tree 6.17 `amdxdna` predates**. The in-tree driver
returns `EOPNOTSUPP` (95) / `EINVAL` (-22) for these requests rather than servicing them.

## Evidence the staging driver fixes it

The `amd/xdna-driver` staging build exposes the symbols the SHIM needs:

```
$ strings build/Release/bins/driver/amdxdna.ko | grep -iE 'get_array|telemetry'
amdxdna_get_telemetry
aie2_query_telemetry
aie2_get_array
```

After `rmmod amdxdna && insmod build/Release/bins/driver/amdxdna.ko` (vermagic matches
`6.17.0-6-generic`), the same `shim_test` run:

```
====== 65: query telemetry passed  =====
====== 66: query telemetry header-only buffer fails passed  =====
```

`shim_test` result on staging: **32 subtests pass** (telemetry/get_array now succeed).

## Impact

Anyone running a current XRT/IRON userspace against the **mainline** `amdxdna` on a Ryzen
7040/8040 (XDNA1) Linux box silently loses telemetry/array functionality and must build +
side-load the out-of-tree driver — a non-obvious step that isn't surfaced by the error
(`Operation not supported` doesn't point at a version mismatch). This is a real adoption
papercut for the gen-1 NPU on Linux.

## Ask

1. **Backport** `GET_ARRAY` + AIE2 telemetry ioctls to the in-tree/staging `amdxdna`, or
2. **Document a minimum in-tree driver version** for each XRT SHIM release, and
3. Have XRT **degrade gracefully** (warn, don't hard-fail) when a driver lacks an optional
   query ioctl, so basic NPU use still works on older in-tree drivers.

## Reproduction

```bash
# build XRT + SHIM + staging driver per amd/xdna-driver README, then:
source /opt/xilinx/xrt/setup.sh
# (A) mainline driver loaded:
sudo modprobe amdxdna
./build/Release/bins/bin/shim_test.sh    # telemetry/get_array subtests FAIL (EOPNOTSUPP/EINVAL)
# (B) staging driver:
sudo modprobe -r amdxdna
sudo insmod build/Release/bins/driver/amdxdna.ko
./build/Release/bins/bin/shim_test.sh    # telemetry/get_array subtests PASS
```

*Reported from the [open-xdna](https://github.com/Scottcjn/open-xdna) gen-1 XDNA1 bring-up,
Elyan Labs.*
