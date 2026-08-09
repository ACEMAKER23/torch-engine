#!/bin/bash
# Profiles every matmul kernel variant with Nsight Compute (ncu) using the
# lightweight matmul_profile harness (one/two kernel launches per shape, so
# ncu doesn't have to instrument a full benchmark sweep).
#
# Usage (from the build/ directory, after building the matmul_profile target):
#   sudo bash ../src/benchmarks/run_ncu_sweep.sh
#
# Output: one .ncu-rep report per (kernel, shape) pair under ./ncu_reports/

set -e
NCU=${NCU:-/usr/local/cuda-13.2/bin/ncu}
BIN=./matmul_profile
OUTDIR=./ncu_reports
mkdir -p "$OUTDIR"

if [ ! -x "$BIN" ]; then
  echo "Error: $BIN not found. Run this script from the build/ directory." >&2
  exit 1
fi

# name M K N
KERNELS=(
  "naive 512 512 512"
  "shared_memory 512 512 512"
  "register_blocking 1024 1024 1024"
  "vectorized_input 1024 1024 1024"
  "warp_tiling 1024 1024 1024"
  "double_buffered 1024 1024 1024"
  "double_buffered_cpasync 1024 1024 1024"
  "double_buffered_swizzled 1024 1024 1024"
  "vector_storage 1024 1024 1024"
  "3stage_cpasync 1024 1024 1024"
  "cublas_f32 1024 1024 1024"
  "tensor_core 1024 1024 1024"
  "ampere 1024 1024 1024"
  "tensor_core_padded 1000 1000 1000"
  "ampere_padded 1000 1000 1000"
  "cublas_tc 1024 1024 1024"
)

for entry in "${KERNELS[@]}"; do
  read -r name M K N <<< "$entry"
  echo "=== Profiling $name ($M x $K x $N) ==="
  "$NCU" --set full -k regex:"matmul|gemm|Gemm|Sgemm" -c 2 -f \
    -o "$OUTDIR/${name}_${M}x${K}x${N}" \
    "$BIN" "$name" "$M" "$K" "$N" || echo "  (failed, continuing)"
done

echo "Done. Reports written to $OUTDIR/*.ncu-rep"
echo "Summarize with e.g.: $NCU --import $OUTDIR/warp_tiling_1024x1024x1024.ncu-rep --page details"
