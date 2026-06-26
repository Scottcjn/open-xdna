# Power-aware NPU instrumentation (DPM level + RAPL)

`examples/npu_powerlog.py` pairs the XDNA1 NPU's **DPM power-state** (from the
staging `amdxdna` debugfs nodes) with **real package power** (RAPL), as an honest
timeline. It turns "the NPU has a low power floor" from an assertion into a
measurement — within the limits of what the hardware actually exposes.

## What is and isn't measured

- **NPU side** — the active **DPM level** index and its `(npuclk, hclk)` clocks,
  plus the SMU `powerstate`, from `/sys/kernel/debug/accel/<bdf>/`. This is a
  clock-state / level signal (L0 lowest power … Lmax highest). It is **not watts**,
  and the two columns are **clocks, not freq+voltage**.
- **Power side** — real watts from RAPL (`/sys/class/powercap/intel-rapl:*`) as
  dEnergy/dt: `package-0` (whole SoC) and `core` (x86 cores). There is **no
  NPU-specific RAPL domain**, so NPU power is not directly isolable. We report
  `rest = package − core` (uncore + iGPU + NPU + fabric) and the idle→active
  **delta**, never a fabricated per-NPU wattage.

Requires root (debugfs + RAPL are root-only) and the staging `amdxdna.ko`
(mainline/DKMS exports neither node — see amd/xdna-driver#1447).

## Measured result (Ryzen 7 8845HS, RyzenAI-npu1, 512³ int16 matmul loop)

5 Hz, 4 s idle baseline → NPU matmul workload → 4 s idle tail (104 samples):

| State | DPM level (mean) | package-0 | core | rest (pkg−core) |
|---|---|---|---|---|
| Idle | 0.0 (L0) | 24.20 W | 2.90 W | 21.30 W |
| NPU workload | 4.38 | 27.08 W | 2.97 W | 24.10 W |
| **Delta** | **0 → 4.38** | **+2.88 W** | **+0.07 W** | **+2.81 W** |

**Reading:** running the NPU raises package power ~2.9 W, and **~98% of that lands
in `rest` (+2.81 W), not the x86 cores (+0.07 W)** — strong evidence the work is
genuinely on the NPU and the host barely spends CPU feeding it. The DPM level
rises L0→~L4 in lockstep. So the NPU subsystem draw under this load is bounded at
**~2.8 W over idle** — measured, honestly scoped (it's the package delta for
NPU + fabric, an upper bound on the NPU alone), and corroborated from both sides
(RAPL watts + NPU DPM level).

This is consistent with the project's low-power-floor thesis without overclaiming
an isolated NPU wattage the hardware does not expose.

## Usage

```bash
# Timeline around a workload, with idle baseline + tail and an auto idle/workload summary:
sudo python3 examples/npu_powerlog.py --hz 5 --warmup 4 \
  --mark-cmd "python3 examples/npu_tiny_mlp.py" --out run.jsonl

# Free-running timeline for N seconds:
sudo python3 examples/npu_powerlog.py --hz 5 --duration 30 --out run.jsonl
```

`--mark-cmd` is parsed as argv (`shlex.split`, no shell): shell features (pipes,
`&&`, `$VAR`, redirects, globs) are not supported — wrap them in a script.

Each JSONL row: `t`, `mark` (idle/workload), `dpm_level`, `npuclk_mhz`,
`hclk_mhz`, `powerstate`, `power_w{package-0, core, rest}`. A summary
(idle vs workload means + deltas) prints to stderr at the end.

## Limits / honest frontier

- **Upper bound, not isolation.** `rest` includes iGPU + uncore + fabric. The
  clean attribution here relies on core power staying flat (it did: +0.07 W),
  which says the host wasn't doing the work — but it is not a dedicated NPU rail.
- **No absolute NPU watts.** The `telemetry_health` buffer (which would carry
  real power) reads `0xff` with production firmware; `xrt-smi` "Estimated Power"
  is `N/A`. Real per-NPU watts need the `QUERY_TELEMETRY` path enabled (likely
  dev firmware) or AMD wiring it (#1447).
- For tighter bounds, run a CPU-only control of the same arithmetic and subtract.
