#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Scott Boudreaux / Elyan Labs. Commercial license: see ../COMMERCIAL.md
#
# open-xdna :: npu_powerlog — pair the XDNA1 NPU DPM power-state (from the staging
# amdxdna debugfs nodes) with real package power (RAPL), as an honest timeline.
#
# WHAT IT MEASURES (and what it does NOT):
#   * NPU side  — the active DPM *level* and its (npuclk, hclk) clocks, plus the
#     SMU powerstate, read from /sys/kernel/debug/accel/<bdf>/{dpm_level,powerstate}.
#     This is a clock-state / level signal (L0 lowest power .. Lmax highest).
#     It is NOT watts, and the columns are two CLOCKS, not freq+voltage.
#   * Power side — real watts from RAPL (/sys/class/powercap/intel-rapl:*),
#     computed as dEnergy/dt: `package-0` (whole SoC) and `core` (x86 cores).
#     There is NO NPU-specific RAPL domain, so NPU power is not directly isolable.
#     We report `rest = package - core` (uncore + iGPU + NPU + fabric) and the
#     idle->active DELTA, never a fabricated per-NPU wattage.
#
# Honest reading: when an NPU workload runs, DPM climbs toward Lmax and the
# package/rest power rises above the idle baseline. The host feeding the NPU also
# burns core power, so the package delta is an upper bound on the NPU's draw, not
# the NPU's draw alone. Pair with a CPU-only control run to bound the host share.
#
# Requires root (debugfs + RAPL are root-only) and the staging amdxdna.ko loaded
# (mainline/DKMS exports neither dpm_level nor powerstate). See amd/xdna-driver#1447.
#
# Usage:
#   sudo python3 examples/npu_powerlog.py --hz 5 --duration 30 --out /tmp/run.jsonl
#   sudo python3 examples/npu_powerlog.py --mark-cmd "python3 examples/npu_tiny_mlp.py"
#
import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

DEBUGFS_ACCEL = "/sys/kernel/debug/accel"
RAPL_ROOT = "/sys/class/powercap"
_DPM_TOKEN = re.compile(r"(?P<lb>\[?)\s*(?P<npuclk>\d+)\s*,\s*(?P<hclk>\d+)\s*(?P<rb>\]?)")


# ---------- NPU DPM (clock-state level) ----------

def find_accel_dir() -> str | None:
    try:
        subs = sorted(d for d in glob.glob(f"{DEBUGFS_ACCEL}/*") if os.path.isdir(d))
    except OSError:
        return None
    return subs[0] if subs else None


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return None


def parse_dpm(text: str) -> dict | None:
    """Active DPM level index + (npuclk_mhz, hclk_mhz). Both are CLOCKS, not voltage."""
    levels, active = [], None
    for m in _DPM_TOKEN.finditer(text):
        s = {"index": len(levels), "npuclk_mhz": int(m.group("npuclk")),
             "hclk_mhz": int(m.group("hclk"))}
        if m.group("lb") == "[" or m.group("rb") == "]":
            active = s
        levels.append(s)
    if not levels:
        return None
    return {"active": active, "max_index": levels[-1]["index"], "levels": len(levels)}


def read_npu(accel_dir: str | None) -> dict:
    if not accel_dir:
        return {"available": False, "reason": "no_accel_debugfs"}
    dpm_raw = _read(f"{accel_dir}/dpm_level")
    ps = _read(f"{accel_dir}/powerstate")
    if dpm_raw is None and ps is None:
        return {"available": False, "reason": "debugfs_unreadable"}
    out = {"available": True, "powerstate": ps, "dpm": parse_dpm(dpm_raw or "")}
    return out


# ---------- RAPL (real watts via dEnergy/dt) ----------

def discover_rapl() -> dict[str, dict]:
    """Map domain name -> {energy_path, max_uj}. package-0, core, etc."""
    domains = {}
    for d in sorted(glob.glob(f"{RAPL_ROOT}/intel-rapl:*")):
        name = _read(f"{d}/name")
        epath = f"{d}/energy_uj"
        if name and os.path.exists(epath):
            mx = _read(f"{d}/max_energy_range_uj")
            domains[name] = {"energy_path": epath, "max_uj": int(mx) if mx else None}
    return domains


def read_energy(domains: dict) -> dict[str, int]:
    out = {}
    for name, info in domains.items():
        v = _read(info["energy_path"])
        if v is not None:
            out[name] = int(v)
    return out


def delta_uj(prev: int, cur: int, max_uj: int | None) -> int:
    """Handle RAPL counter wraparound."""
    if cur >= prev:
        return cur - prev
    return (max_uj - prev + cur) if max_uj else 0  # wrapped


def main() -> int:
    ap = argparse.ArgumentParser(description="NPU DPM power-state + RAPL package power timeline")
    ap.add_argument("--hz", type=float, default=5.0, help="samples per second")
    ap.add_argument("--duration", type=float, default=30.0, help="seconds (ignored if --mark-cmd)")
    ap.add_argument("--out", type=str, default=None, help="JSONL output path (default stdout)")
    ap.add_argument("--mark-cmd", type=str, default=None,
                    help="run this command mid-capture and mark its window")
    ap.add_argument("--warmup", type=float, default=4.0, help="idle baseline seconds before --mark-cmd")
    args = ap.parse_args()

    if os.geteuid() != 0:
        print("npu_powerlog needs root (debugfs + RAPL are root-only)", file=sys.stderr)
        return 2

    accel = find_accel_dir()
    domains = discover_rapl()
    if not accel:
        print("WARN: no NPU debugfs (staging amdxdna.ko not loaded?) — logging RAPL only", file=sys.stderr)
    if not domains:
        print("WARN: no RAPL domains found — logging NPU state only", file=sys.stderr)

    out_fh = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    interval = 1.0 / args.hz
    samples = []

    def sample(prev_e, prev_t, mark):
        now = time.monotonic()
        cur_e = read_energy(domains)
        dt = now - prev_t if prev_t else None
        power = {}
        if dt and prev_e:
            for name in cur_e:
                if name in prev_e:
                    du = delta_uj(prev_e[name], cur_e[name], domains[name]["max_uj"])
                    power[name] = round(du / 1e6 / dt, 3)  # uJ -> W
            if "package-0" in power and "core" in power:
                power["rest"] = round(power["package-0"] - power["core"], 3)
        npu = read_npu(accel)
        act = (npu.get("dpm") or {}).get("active") if npu.get("available") else None
        row = {
            "t": round(now, 4), "mark": mark,
            "dpm_level": act["index"] if act else None,
            "npuclk_mhz": act["npuclk_mhz"] if act else None,
            "hclk_mhz": act["hclk_mhz"] if act else None,
            "powerstate": npu.get("powerstate"),
            "power_w": power or None,
        }
        out_fh.write(json.dumps(row) + "\n")
        out_fh.flush()
        samples.append(row)
        return cur_e, now

    prev_e, prev_t = read_energy(domains), time.monotonic()
    time.sleep(interval)

    proc = None
    if args.mark_cmd:
        # idle baseline
        t_end = time.monotonic() + args.warmup
        while time.monotonic() < t_end:
            prev_e, prev_t = sample(prev_e, prev_t, "idle")
            time.sleep(interval)
        # launch workload, mark its window
        import subprocess
        proc = subprocess.Popen(args.mark_cmd, shell=True)
        while proc.poll() is None:
            prev_e, prev_t = sample(prev_e, prev_t, "workload")
            time.sleep(interval)
        # idle tail
        t_end = time.monotonic() + args.warmup
        while time.monotonic() < t_end:
            prev_e, prev_t = sample(prev_e, prev_t, "idle")
            time.sleep(interval)
    else:
        t_end = time.monotonic() + args.duration
        while time.monotonic() < t_end:
            prev_e, prev_t = sample(prev_e, prev_t, None)
            time.sleep(interval)

    if args.out:
        out_fh.close()

    # ---- summary (idle vs workload) ----
    def avg(rows, key, sub=None):
        vals = []
        for r in rows:
            v = r.get("power_w") if sub else r.get(key)
            if sub and isinstance(v, dict):
                v = v.get(sub)
            if isinstance(v, (int, float)):
                vals.append(v)
        return round(sum(vals) / len(vals), 3) if vals else None

    idle = [r for r in samples if r.get("mark") == "idle"]
    work = [r for r in samples if r.get("mark") == "workload"]
    summary = {"samples": len(samples)}
    if idle or work:
        for grp, rows in (("idle", idle), ("workload", work)):
            if not rows:
                continue
            summary[grp] = {
                "n": len(rows),
                "pkg_w": avg(rows, "power_w", "package-0"),
                "core_w": avg(rows, "power_w", "core"),
                "rest_w": avg(rows, "power_w", "rest"),
                "dpm_level_mean": avg(rows, "dpm_level"),
            }
        if "idle" in summary and "workload" in summary:
            i, w = summary["idle"], summary["workload"]
            if i.get("pkg_w") is not None and w.get("pkg_w") is not None:
                summary["delta_pkg_w"] = round(w["pkg_w"] - i["pkg_w"], 3)
            if i.get("rest_w") is not None and w.get("rest_w") is not None:
                summary["delta_rest_w"] = round(w["rest_w"] - i["rest_w"], 3)
    print("\n# SUMMARY", json.dumps(summary, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
