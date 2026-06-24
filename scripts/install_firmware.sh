#!/usr/bin/env bash
# open-xdna :: install/match XDNA1 (device 1502) NPU firmware.
# Stale firmware causes ERT_CMD_STATE_ABORT / mailbox ret -22 on command submission.
#   sudo bash scripts/install_firmware.sh [version]   (default 1.5.5.391)
# Reloads the staging driver afterward so the new firmware is loaded.
set -uo pipefail
VER="${1:-1.5.5.391}"
KO="${KO:-$HOME/xdna-driver/build/Release/bins/driver/amdxdna.ko}"
FW=/lib/firmware/amdnpu/1502_00
URL="https://gitlab.com/kernel-firmware/drm-firmware/-/raw/amd-ipu-staging/amdnpu/1502_00/npu.sbin.${VER}"

mkdir -p "$FW"
echo "downloading npu.sbin.$VER ..."
curl -fL -o "$FW/npu.sbin.$VER" "$URL" || { echo "download failed: $URL"; exit 1; }
echo "  $(stat -c %s "$FW/npu.sbin.$VER") bytes"

# back up any existing compressed firmware, point npu.sbin at the new uncompressed blob
[ -e "$FW/npu.sbin.zst" ] && cp -n "$FW/npu.sbin.zst" "$FW/npu.sbin.zst.bak" 2>/dev/null || true
ln -sf "npu.sbin.$VER" "$FW/npu.sbin"
rm -f "$FW/npu.sbin.zst"          # remove stale compressed so the loader uses ours
ls -l "$FW"

echo "reloading driver to pick up firmware ..."
modprobe -r amdxdna 2>/dev/null || rmmod amdxdna 2>/dev/null || true
if [ -f "$KO" ]; then insmod "$KO"; else modprobe amdxdna; fi
sleep 2
dmesg 2>/dev/null | grep -iE "Load firmware|Initialized amdxdna" | tail -2
