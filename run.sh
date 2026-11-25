#!/bin/bash
#SBATCH --job-name=dsu-starter
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=dsu_%j.out
#SBATCH --error=dsu_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "======================================"
echo "Dynamic Graph DSU - CUDA starter"
echo "Job ID : ${SLURM_JOB_ID:-interactive}"
echo "Node   : $(hostname)"
echo "Date   : $(date)"
echo "======================================"

module purge
module load cuda/11.3

echo
echo "===== GPU ====="
nvidia-smi

echo
echo "===== CUDA compiler ====="
command -v nvcc
nvcc --version

echo
echo "===== Host compiler ====="
HOST_CXX="$(command -v g++ || true)"
if [[ -z "${HOST_CXX}" ]]; then
    echo "ERROR: g++ was not found in the environment."
    exit 1
fi
echo "Using: ${HOST_CXX}"
"${HOST_CXX}" --version

# Check the system C library development header needed by CUDA's host compiler.
if [[ ! -f /usr/include/features.h ]]; then
    echo "ERROR: /usr/include/features.h is missing on this compute node."
    echo "This is a system development-header/toolchain issue, not a DSU source issue."
    echo "Ask the cluster administrator to provide glibc development headers"
    echo "or a compatible GCC toolchain/module."
    exit 1
fi

[[ -f dsu.cu ]] || { echo "ERROR: dsu.cu not found"; exit 1; }
[[ -f sample_graph.txt ]] || { echo "ERROR: sample_graph.txt not found"; exit 1; }

# A100 = sm_80. For H100, use sm_90.
GPU_ARCH="${GPU_ARCH:-sm_80}"

echo
echo "===== Compile for ${GPU_ARCH} ====="
nvcc -ccbin "${HOST_CXX}" -O2 -std=c++17 \
     -arch="${GPU_ARCH}" dsu.cu -o dsu

echo
echo "===== Run ====="
./dsu sample_graph.txt

echo
echo "Completed."
