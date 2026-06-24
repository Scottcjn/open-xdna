#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Scott Boudreaux / Elyan Labs. Commercial license: see COMMERCIAL.md
# open-xdna :: run an IRON programming example on the XDNA1 NPU (local JIT compile + run)
#
# Usage:  sudo bash scripts/run_example.sh <example-relpath>
#   e.g.  sudo bash scripts/run_example.sh basic/passthrough_kernel
#         sudo bash scripts/run_example.sh basic/matrix_multiplication/single_core
#
# NOTE: deliberately does NOT use `set -e` — sourcing XRT/IRON env scripts can return
# nonzero and would silently abort the run under `set -e`.
MLIR_AIE_DIR="${MLIR_AIE_DIR:-$HOME/open-xdna/mlir-aie}"
LLVM_BIN="${LLVM_BIN:-/usr/lib/llvm-20/bin}"   # provides llvm-objcopy (IRON needs it on PATH)
XRT_SETUP="${XRT_SETUP:-/opt/xilinx/xrt/setup.sh}"
EX="${1:?usage: run_example.sh <path under mlir-aie/programming_examples/>}"

export PATH="$LLVM_BIN:$PATH"
# shellcheck disable=SC1090
source "$XRT_SETUP" >/dev/null 2>&1
cd "$MLIR_AIE_DIR" || { echo "mlir-aie not found at $MLIR_AIE_DIR (run setup_iron.sh)"; exit 1; }
# shellcheck disable=SC1091
source ironenv/bin/activate
# shellcheck disable=SC1091
source utils/env_setup.sh >/dev/null 2>&1

echo "llvm-objcopy : $(command -v llvm-objcopy)"
echo "aie-opt      : $(command -v aie-opt)"
echo "PEANO        : $PEANO_INSTALL_DIR"
echo "--- example  : $EX ---"
cd "programming_examples/$EX" || { echo "no such example: $EX"; exit 1; }
make run
