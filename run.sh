#!/bin/bash
# run.sh : SLURM job script for DynamicGraphSimplified on Bhavani cluster
#
# Before submitting, compile once on the master/login node:
#   module purge
#   module load cuda/11.3
#   make SM=80
#
# Then submit:
#   sbatch run.sh
#
#SBATCH --job-name=dyn_graph
#SBATCH --partition=normal
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -e

# Configurable CUDA module (default: cuda/11.3 on Bhavani)
CUDA_MODULE="${CUDA_MODULE:-cuda/11.3}"

echo "=============================="
echo "Job ID       : ${SLURM_JOB_ID:-interactive}"
echo "Node         : $(hostname)"
echo "Date         : $(date)"
echo "=============================="

if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/lmod.sh /opt/ohpc/admin/lmod/lmod/init/bash /etc/profile.d/modules.sh; do
        [ -f "$f" ] && source "$f" && break
    done
fi

module purge
module load "$CUDA_MODULE"
echo "Loaded module: $CUDA_MODULE"

echo ""
echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
echo ""

# Verify pre-built binary (compute nodes may lack glibc/C++ headers for nvcc)
if [ ! -x "./dynamic_graph" ]; then
    echo "ERROR: ./dynamic_graph not found."
    echo "Compile on the master/login node first:"
    echo "  module purge && module load $CUDA_MODULE && make SM=80"
    exit 1
fi

mkdir -p results logs

# ============================================================
# 1. Small Correctness Tests (stdout only, no result files)
# ============================================================
echo ">>> Correctness Test 1: DSU (data/dsu_test.txt)"
./dynamic_graph dsu data/dsu_test.txt \
    --init-frac 0.5 --batch-size 2 --num-batches -1 --verify-every 1 --no-out
DSU_STATUS=$?
echo ""

echo ">>> Correctness Test 2: SSSP (data/sssp_test.txt)"
./dynamic_graph sssp data/sssp_test.txt \
    --source 0 --init-frac 0.5 --batch-size 2 --num-batches -1 --verify-every 1 --no-out
SSSP_STATUS=$?
echo ""

# ============================================================
# 2. Wiki-Vote Scalability Tests (saved to 2 separate .txt files)
# ============================================================
if [ -f "data/wiki-Vote.txt" ]; then
    echo ">>> Wiki-Vote Benchmark 1: DSU -> results/wiki_dsu.txt"
    ./dynamic_graph dsu data/wiki-Vote.txt \
        --directed --init-frac 0.7 --batch-size 1000 --num-batches 20 \
        --verify-every 1 --baseline-every 1 --out results/wiki_dsu.txt
    echo ""

    echo ">>> Wiki-Vote Benchmark 2: SSSP -> results/wiki_sssp.txt"
    ./dynamic_graph sssp data/wiki-Vote.txt \
        --directed --init-frac 0.7 --batch-size 1000 --num-batches 10 \
        --verify-every 1 --baseline-every 1 --out results/wiki_sssp.txt
    echo ""
else
    echo "WARNING: data/wiki-Vote.txt not found — skipping Wiki-Vote tests."
fi

# ============================================================
# 3. Large-Scale GPU Speedup Benchmark (R-MAT: 1M nodes, 16.7M edges)
# ============================================================
echo ">>> Large Benchmark 1: DSU (R-MAT scale=20, 16.7M edges, 500k/batch) -> results/large_dsu.txt"
./dynamic_graph --mode cc --graph rmat --scale 20 --edge-factor 16 \
    --init-frac 0.3 --batch-size 500000 --num-batches 10 \
    --verify-every 1 --baseline-every 1 --out results/large_dsu.txt
echo ""

echo ">>> Large Benchmark 2: SSSP (R-MAT scale=20, 16.7M edges, 100k/batch) -> results/large_sssp.txt"
./dynamic_graph --mode sssp --graph rmat --scale 20 --edge-factor 16 \
    --init-frac 0.7 --batch-size 100000 --num-batches 10 \
    --verify-every 1 --baseline-every 1 --no-bf --out results/large_sssp.txt
echo ""

echo "=============================="
echo "All tests complete."
echo "DSU correctness : exit $DSU_STATUS"
echo "SSSP correctness: exit $SSSP_STATUS"
echo "Saved reports   : results/wiki_dsu.txt, results/wiki_sssp.txt, results/large_dsu.txt, results/large_sssp.txt"
echo "Date: $(date)"
echo "=============================="

[ $DSU_STATUS -eq 0 ] && [ $SSSP_STATUS -eq 0 ]
