#!/usr/bin/env bash
# open-xdna :: load the matched (staging) amdxdna.ko built from amd/xdna-driver.
# Fixes mainline's missing aie2_query_telemetry / aie2_get_array ioctls.
#   sudo bash scripts/swap_driver.sh [path/to/amdxdna.ko]
# Reverts on reboot (mainline reloads). Restore now with: modprobe amdxdna
set -uo pipefail
KO="${1:-$HOME/xdna-driver/build/Release/bins/driver/amdxdna.ko}"
[ -f "$KO" ] || { echo "driver .ko not found: $KO  (build amd/xdna-driver first)"; exit 1; }

want=$(modinfo -F vermagic "$KO" | awk '{print $1}')
have=$(uname -r)
[ "$want" = "$have" ] || echo "WARN: .ko vermagic ($want) != running kernel ($have)"

echo "unloading current amdxdna ..."
modprobe -r amdxdna 2>/dev/null || rmmod amdxdna 2>/dev/null || true
echo "inserting $KO ..."
insmod "$KO" || { echo "insmod failed; restoring mainline"; modprobe amdxdna; exit 1; }
sleep 2
echo "loaded srcversion: $(cat /sys/module/amdxdna/srcversion 2>/dev/null)"
[ -e /dev/accel/accel0 ] && echo "OK: /dev/accel/accel0 present" || echo "WARN: no accel node"
