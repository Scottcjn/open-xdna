#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Scott Boudreaux / Elyan Labs. Commercial license: see COMMERCIAL.md
#
# open-xdna :: IRON / mlir-aie setup (Phase A — the network part)
# Run this in YOUR shell (it needs internet):   ! bash ~/open-xdna/setup_iron.sh
#
# Target: AMD XDNA1 / Phoenix-Hawk-Point NPU (RyzenAI-npu1) on Ubuntu 25.10.
# Installs the open-source IRON framework + Peano (llvm-aie) AIE compiler via wheels.
# Python 3.13 is supported by the wheels per the mlir-aie README.
#
set -euo pipefail
cd ~/open-xdna

echo "==> [1/6] Clone mlir-aie (IRON framework + programming examples)"
if [ ! -d mlir-aie/.git ]; then
  git clone https://github.com/Xilinx/mlir-aie.git
else
  echo "    already cloned; pulling latest"
  git -C mlir-aie pull --ff-only || true
fi
cd mlir-aie
echo "    mlir-aie commit: $(git rev-parse --short HEAD)"

echo "==> [2/6] Create Python 3.13 venv (ironenv)"
if [ ! -d ironenv ]; then
  python3.13 -m venv ironenv
fi
# shellcheck disable=SC1091
source ironenv/bin/activate
python3 -m pip install --upgrade pip wheel

echo "==> [3/6] Install IRON wheel (mlir_aie, latest-wheels-4)"
python3 -m pip install mlir_aie \
  -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-4

echo "==> [4/6] Install Peano / llvm-aie (AIE LLVM backend, nightly)"
python3 -m pip install llvm-aie \
  -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly

echo "==> [5/6] Install example python requirements"
[ -f python/requirements.txt ] && python3 -m pip install -r python/requirements.txt || true
[ -f programming_examples/requirements.txt ] && python3 -m pip install -r programming_examples/requirements.txt || true

echo "==> [6/6] Sanity check: import aie + locate peano"
python3 - <<'PY'
import importlib, sys
ok = True
try:
    import aie
    print("    aie module OK ->", aie.__file__)
except Exception as e:
    print("    aie import FAILED:", e); ok = False
try:
    import aie.iron as iron  # noqa
    print("    aie.iron OK")
except Exception as e:
    print("    aie.iron import note:", e)
try:
    import llvmaie  # peano wheel python shim (name varies)
    print("    llvmaie OK")
except Exception:
    pass
sys.exit(0 if ok else 1)
PY

echo ""
echo "============================================================"
echo " Phase A done. mlir-aie + peano installed in:"
echo "   ~/open-xdna/mlir-aie/ironenv"
echo " Peano (llvm-aie) wheel files:"
python3 -c "import importlib.util,os; s=importlib.util.find_spec('llvm_aie') or importlib.util.find_spec('llvmaie'); print('  ', os.path.dirname(s.origin) if s else 'pip show llvm-aie for path')" 2>/dev/null || true
pip show llvm-aie 2>/dev/null | sed -n 's/^Location: /  llvm-aie at: /p'
echo ""
echo " NEXT: tell Claude 'Phase A done' so it can read env_setup.sh,"
echo "       configure the npu1 matmul example, and wire it to the NPU."
echo "============================================================"
