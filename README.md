# DynamicGraphSimplified

GPU-accelerated Dynamic Graph Analysis — Disjoint Set Union (DSU) and Single-Source Shortest Path (SSSP).

## Project Structure

```
DynamicGraphSimplified/
├── main.cu              # Entry point: CLI parsing, batch loop, verification, CSV output
├── dsu.cu               # GPU DSU + GPU DeviceDynGraph (device_mem_stats, grid_for defined here)
├── sssp.cu              # GPU SSSP (frontier-based, Bellman-Ford, dynamic incremental)
├── common.cuh           # Consolidated header: types, macros, class declarations
├── cpu_reference.cpp    # CPU DSU, Dijkstra, graph generators, SNAP/DIMACS loaders
│
├── Makefile             # Builds dynamic_graph; supports SM= override
├── run.sh               # SLURM batch script: build + correctness + scalability tests
├── README.md
│
├── data/
│   ├── dsu_test.txt     # Correctness test for DSU (10 vertices, 2 components)
│   ├── sssp_test.txt    # Correctness test for SSSP (7 vertices, known distances)
│   └── wiki-Vote.txt    # SNAP Wiki-Vote dataset (scalability testing)
│
└── results/             # CSV output from runs
```

## Algorithms

### GPU Disjoint Set Union (DSU)
Maintains **weakly connected components** of the dynamic graph as edge insertions arrive.

- Each vertex stores `[rank:32 | parent:32]` as one 64-bit word.
- Lock-free union-by-rank with a single `atomicCAS` per link.
- Path-halving during `find`; full path compression after each batch.
- After each batch, the component count is updated with warp-reduced `atomicSub`.
- Directed edges are treated as undirected connections for WCC.
- **Note:** standard DSU does not support arbitrary deletions. Use `recompute_from_graph` for full recomputation if needed.

### GPU SSSP (Single-Source Shortest Path)
Computes and maintains shortest-path distances from a fixed source as edges are inserted.

- Distances stored as `[dist:32 | pred:32]` (64-bit), updated atomically with `atomicMin`.
- **Static baselines:** frontier-based BFS-style propagation and topology-driven Bellman-Ford.
- **Dynamic update:** seed kernel tests each inserted arc; improved destinations seed the frontier. Only the affected subgraph is re-expanded.
- **Load-balanced propagation:** vertices classified into small/medium/large degree bins; thread-, warp-, and block-granularity kernels handle each bin.
- Negative-weight edges are **not** supported.

## Quickstart (on your GPU cluster)

### 1. Build (on the master/login node)

Some HPC clusters omit C/C++ development headers (`<features.h>`, `<chrono>`) on compute nodes, so compile on the **master/login node** before submitting a SLURM job:

```bash
module purge
module load cuda/11.3          # check exact CUDA module via `module avail cuda`
make SM=80                     # sm_80 = NVIDIA A30/A100 (70=V100, 75=T4, 86=RTX3090)
```

### 2. Correctness Tests

```bash
# DSU: 10-vertex graph → expects 2 components: {0..4} and {5..9}
./dynamic_graph dsu data/dsu_test.txt \
    --init-frac 0.5 --batch-size 2 --num-batches -1 --verify-every 1 --no-out

# SSSP: 7-vertex weighted graph → expects distances [0,2,5,7,6,INF,INF] from vertex 0
./dynamic_graph sssp data/sssp_test.txt \
    --source 0 --init-frac 0.5 --batch-size 2 --num-batches -1 --verify-every 1 --no-out
```

### 3. Scalability Tests (Wiki-Vote real-world dataset)

```bash
./dynamic_graph dsu data/wiki-Vote.txt \
    --directed --init-frac 0.7 --batch-size 1000 --num-batches 20 \
    --out results/wiki_dsu.txt

./dynamic_graph sssp data/wiki-Vote.txt \
    --directed --init-frac 0.7 --batch-size 1000 --num-batches 10 \
    --out results/wiki_sssp.txt
```

### 4. SLURM Submission

Compile on `master`, then submit:

```bash
sbatch run.sh
```

## Command Reference

```
./dynamic_graph [dsu|sssp] FILE [options]
./dynamic_graph [options]

Positional commands:
  dsu  FILE    Run DSU (connected components) only on FILE
  sssp FILE    Run SSSP only on FILE

Graph source:
  --graph uniform|rmat|grid|snap|dimacs   (default: uniform)
  --file PATH          SNAP/DIMACS file path
  --directed           treat graph as directed
  --n N --m M          uniform graph parameters
  --scale S --edge-factor E   R-MAT parameters

Update stream:
  --init-frac F        fraction of edges in initial graph (default: 0.5)
  --batch-size B       insertions per batch (default: 1000)
  --num-batches K      number of batches; -1 = whole stream (default: 10)

Algorithms:
  --mode all|cc|sssp   which algorithms to run (default: all)
  --source S           SSSP source vertex; -1 = max-degree vertex (default: -1)
  --no-lb              disable degree-aware load balancing
  --block T            CUDA threads per block (default: 256)
  --merge-frac F       merge delta into base when |delta| > F*|base| (default: 0.1)
  --queries Q          connectivity queries per batch (default: 1000)

Measurement:
  --verify-every K     verify every K batches; 0 = disabled (default: 1)
  --baseline-every K   run static baselines every K batches (default: 1)
  --no-bf              skip Bellman-Ford baseline
  --out PATH           write formatted .txt table report to PATH
  --no-out             do not write any .txt file (stdout only)
  --csv PATH           optional compact CSV output path
  --tag NAME           label for CSV rows (default: run)
  --seed S             RNG seed (default: 42)
  --device D           CUDA device index (default: 0)
  --quiet              suppress per-batch output
```

## Test File Formats

### DSU test (`data/dsu_test.txt`)

Edge list format: `u v` per line (unweighted), `#` comments ignored.

```
# 10 vertices (0-9), 12 edges → 2 components
0 1
1 2
...
```

After processing all edges: component `{0,1,2,3,4}` and `{5,6,7,8,9}`.

### SSSP test (`data/sssp_test.txt`)

Weighted edge list: `u v weight` per line.

```
0 1 2
0 2 10
...
```

Expected shortest distances from vertex 0:

| Vertex | Distance | Path |
|--------|----------|------|
| 0 | 0 | source |
| 1 | 2 | 0→1 |
| 2 | 5 | 0→1→2 (not direct 10) |
| 3 | 7 | 0→1→3 |
| 4 | 6 | 0→1→2→4 |
| 5 | INF | unreachable |
| 6 | INF | unreachable |

## Real-World Dataset: Wiki-Vote

**Source:** [Stanford SNAP — Wikipedia who-votes-on-whom network](https://snap.stanford.edu/data/wiki-Vote.html)

| Property | Value |
|----------|-------|
| Nodes | 7,115 |
| Directed edges | 103,689 |
| Weights | Assigned deterministically from seed=42 RNG, uniform [1, 100] |
| Format | SNAP edge list: `u v` per line, `#` comments |

**Citation:**  
J. Leskovec, D. Huttenlocher, J. Kleinberg. *Signed Networks in Social Media.* CHI 2010.  
J. Leskovec, D. Huttenlocher, J. Kleinberg. *Predicting Positive and Negative Links in Online Social Networks.* WWW 2010.

The original dataset contains no edge weights. Weights are assigned as:
```
w = uniform_int_distribution<uint32_t>(1, 100)(mt19937_64(seed=42))
```
One weight is drawn per edge in the order they are read from the file. This assignment is deterministic and reproducible.

## Important Notes

- **No local CUDA required:** this project is designed to be compiled and executed on a GPU cluster. All CUDA code must be compiled with `nvcc` on a CUDA-capable machine.
- **DSU insertions only:** the GPU DSU supports dynamic edge *insertions*. Edge deletions require a full recomputation via `--mode cc` with `recompute_from_graph`.
- **SSSP insertions only:** the incremental SSSP update handles edge insertions. For deletions, run a full static recomputation.
- **Directed vs. undirected:** DSU always treats edges as undirected for WCC. SSSP respects `--directed`.

## Build System

The Makefile compiles three CUDA translation units and one C++ file:

| File | Role |
|------|------|
| `main.cu` | Entry point, CLI, batch loop |
| `dsu.cu` | `DeviceDynGraph` + `GpuDSU` + `device_mem_stats()` + `grid_for()` |
| `sssp.cu` | `GpuSSSP` only |
| `cpu_reference.cpp` | CPU algorithms, graph I/O, generators |

`device_mem_stats()` and `grid_for()` are defined **once** in `dsu.cu` and declared `extern` in `common.cuh`.
