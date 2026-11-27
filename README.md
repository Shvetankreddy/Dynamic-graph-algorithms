# Dynamic Graph Algorithms using CUDA

## About the Project

In this project, we use CUDA to process graphs that change over time. We start with an initial graph and add new edges in batches. Instead of running the entire algorithm again after each batch, we update the existing results.

We implemented two algorithms on the GPU and compared their results and execution times with CPU implementations.

## Algorithms

### Disjoint Set Union (DSU)

DSU keeps track of connected groups of vertices. When a new edge connects two different groups, they are merged into one.

For example, if we have two groups {0, 1} and {2, 3}, adding an edge between 1 and 2 combines them into a single group. CUDA allows multiple edges to be processed in parallel.

### Single-Source Shortest Path (SSSP)

SSSP finds the shortest distance from a selected source vertex to all other vertices. When a new edge is added, we check whether it gives a shorter path and update the affected vertices.

For example, if the current distance from A to C is 8 and a new edge connects A to C with a weight of 2, the distance becomes 2.

## How It Works

1. Load a graph and divide its edges into an initial graph and batches of new edges.
2. Calculate the initial connected components or shortest paths.
3. Insert new edges and update the results using CUDA.
4. Compare the GPU results with the CPU results to check correctness and performance.

We use Compressed Sparse Row (CSR) to store the graph efficiently. New edges are stored separately and periodically merged into the main graph.

## Project Structure

| File / Folder | Description |
| --- | --- |
| `main.cu` | Main program and batch processing |
| `dsu.cu` | GPU implementation of DSU and graph storage |
| `sssp.cu` | GPU implementation of SSSP |
| `common.cuh` | Common definitions |
| `cpu_reference.cpp` | CPU implementations and graph loading |
| `Makefile` | Build instructions |
| `run.sh` | Script for running benchmarks on the cluster |
| `data/` | Graph datasets |
| `results/` | Benchmark results |

## How to Run

**Requirements:** CUDA-supported NVIDIA GPU, CUDA Toolkit, C++17 compiler and Make.

Compile the project:

```bash
make
```

Run DSU:

```bash
./dynamic_graph dsu data/dsu_test.txt --init-frac 0.5 --batch-size 2 --num-batches -1 --verify-every 1 --no-out
```

Run SSSP:

```bash
./dynamic_graph sssp data/sssp_test.txt --source 0 --init-frac 0.5 --batch-size 2 --num-batches -1 --verify-every 1 --no-out
```

Here, `--init-frac 0.5` uses half the edges initially, `--batch-size 2` adds two edges per batch, and `--num-batches -1` processes all remaining edges. `--verify-every 1` checks the results after every batch, and `--no-out` skips saving a text report.

For more options, use `./dynamic_graph --help`.

On the Bhavani GPU cluster, the benchmarks can be run using SLURM:

```bash
module purge
module load cuda/11.3
make SM=80
mkdir -p logs 
sbatch run.sh
```

## Datasets and Results

We tested the algorithms using small test graphs, the Wiki-Vote dataset and synthetic graphs such as R-MAT.

The following results are from the saved R-MAT benchmarks on an NVIDIA A30 GPU, using a graph generated with around one million vertices and 16.7 million edges.

| Average time per batch | DSU | SSSP |
| --- | ---: | ---: |
| GPU algorithm | 0.146 ms | 0.900 ms |
| GPU total update | 4.367 ms | 2.305 ms |
| CPU incremental update | 4.2756 ms | 3.6259 ms |
| CPU full Dijkstra | — | 381.2872 ms |

All correctness checks passed in these benchmarks. The GPU algorithm was around 29x faster than CPU DSU updates and 424x faster than full CPU Dijkstra recomputation. These speedups exclude GPU data transfer and graph maintenance. Including those costs, DSU was slightly slower than the CPU incremental version, while SSSP was around 1.57x faster than its CPU incremental version.

## Limitations
The project currently supports edge insertions only, not edge deletions or new vertices. SSSP supports positive edge weights.

## Reference
[Stanford SNAP – Wiki-Vote Dataset](https://snap.stanford.edu/data/wiki-Vote.html)
