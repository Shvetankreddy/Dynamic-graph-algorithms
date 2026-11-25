// Dynamic weakly connected components using CPU DSU and CUDA DSU.
// Directed input edges are treated as undirected for connectivity.
// Vertices are numbered 0 ... N-1.
//
// Input format:
// N M Q
// M initial edges: u v
// Q inserted edges: u v

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

struct Edge {
    int u;
    int v;
};

struct Graph {
    int n = 0;
    std::vector<Edge> initial;
    std::vector<Edge> updates;
};

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << ": " << cudaGetErrorString(err) << '\n'; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

Graph readGraph(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("Cannot open input file: " + filename);
    }

    Graph g;
    int m = 0, q = 0;
    if (!(in >> g.n >> m >> q) || g.n <= 0 || m < 0 || q < 0) {
        throw std::runtime_error("Expected a valid header: N M Q");
    }

    auto readEdges = [&](std::vector<Edge>& edges, int count,
                         const char* description) {
        edges.reserve(count);
        for (int i = 0; i < count; ++i) {
            Edge e{};
            if (!(in >> e.u >> e.v)) {
                throw std::runtime_error(std::string("Missing ") +
                                         description + " edge data");
            }
            if (e.u < 0 || e.u >= g.n || e.v < 0 || e.v >= g.n) {
                throw std::runtime_error("Vertex ID is outside [0, N-1]");
            }
            edges.push_back(e);
        }
    };

    readEdges(g.initial, m, "initial");
    readEdges(g.updates, q, "update");
    return g;
}

class CPU_DSU {
    std::vector<int> parent, size;
    int components;

public:
    explicit CPU_DSU(int n) : parent(n), size(n, 1), components(n) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }

    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (size[a] < size[b]) std::swap(a, b);
        parent[b] = a;
        size[a] += size[b];
        --components;
    }

    void addEdges(const std::vector<Edge>& edges) {
        for (const Edge& e : edges) unite(e.u, e.v);
    }

    int count() const { return components; }

    std::vector<int> labels() {
        std::vector<int> minimum(parent.size(), INT_MAX);
        for (int v = 0; v < static_cast<int>(parent.size()); ++v) {
            minimum[find(v)] = std::min(minimum[find(v)], v);
        }

        std::vector<int> result(parent.size());
        for (int v = 0; v < static_cast<int>(parent.size()); ++v) {
            result[v] = minimum[find(v)];
        }
        return result;
    }
};

__global__ void initializeParents(int* parent, int n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) parent[v] = v;
}

__device__ int findRoot(int* parent, int x) {
    while (true) {
        int p = atomicAdd(&parent[x], 0);
        int gp = atomicAdd(&parent[p], 0);
        if (p == gp) return p;
        x = p;
    }
}

__global__ void connectEdges(int* parent, const Edge* edges,
                             int m, int* changed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    int a = findRoot(parent, edges[i].u);
    int b = findRoot(parent, edges[i].v);

    while (a != b) {
        int high = max(a, b);
        int low = min(a, b);
        int old = atomicMin(&parent[high], low);
        if (old > low) atomicExch(changed, 1);

        a = findRoot(parent, a);
        b = findRoot(parent, b);
    }
}

__global__ void compressParents(int* parent, int n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) parent[v] = findRoot(parent, v);
}

void addEdgesGPU(int* d_parent, const std::vector<Edge>& edges,
                 Edge* d_edges, int* d_changed, int& passes) {
    const int m = static_cast<int>(edges.size());
    if (m == 0) return;

    CUDA_CHECK(cudaMemcpy(d_edges, edges.data(), m * sizeof(Edge),
                          cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    const int blocks = (m + threads - 1) / threads;

    while (true) {
        CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));
        connectEdges<<<blocks, threads>>>(d_parent, d_edges, m, d_changed);
        CUDA_CHECK(cudaGetLastError());
        ++passes;

        int changed = 0;
        CUDA_CHECK(cudaMemcpy(&changed, d_changed, sizeof(int),
                              cudaMemcpyDeviceToHost));
        if (!changed) break;
    }
}

struct GPUResult {
    std::vector<int> labels;
    int components = 0;
    int passes = 0;
};

GPUResult runGPU(const Graph& g, double& elapsedMs) {
    int *d_parent = nullptr, *d_changed = nullptr;
    Edge* d_edges = nullptr;
    const std::size_t capacity =
        std::max<std::size_t>(1, std::max(g.initial.size(), g.updates.size()));

    CUDA_CHECK(cudaMalloc(&d_parent, g.n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edges, capacity * sizeof(Edge)));

    auto start = std::chrono::steady_clock::now();

    constexpr int threads = 256;
    const int blocks = (g.n + threads - 1) / threads;
    initializeParents<<<blocks, threads>>>(d_parent, g.n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GPUResult result;
    addEdgesGPU(d_parent, g.initial, d_edges, d_changed, result.passes);
    addEdgesGPU(d_parent, g.updates, d_edges, d_changed, result.passes);

    compressParents<<<blocks, threads>>>(d_parent, g.n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto stop = std::chrono::steady_clock::now();
    elapsedMs = std::chrono::duration<double, std::milli>(stop - start).count();

    std::vector<int> parent(g.n);
    CUDA_CHECK(cudaMemcpy(parent.data(), d_parent, g.n * sizeof(int),
                          cudaMemcpyDeviceToHost));

    std::vector<int> minimum(g.n, INT_MAX);
    for (int v = 0; v < g.n; ++v) {
        minimum[parent[v]] = std::min(minimum[parent[v]], v);
    }

    result.labels.resize(g.n);
    for (int v = 0; v < g.n; ++v) {
        result.labels[v] = minimum[parent[v]];
    }

    std::vector<int> uniqueLabels = result.labels;
    std::sort(uniqueLabels.begin(), uniqueLabels.end());
    uniqueLabels.erase(std::unique(uniqueLabels.begin(), uniqueLabels.end()),
                       uniqueLabels.end());
    result.components = static_cast<int>(uniqueLabels.size());

    CUDA_CHECK(cudaFree(d_parent));
    CUDA_CHECK(cudaFree(d_changed));
    CUDA_CHECK(cudaFree(d_edges));
    return result;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input.txt>\n";
        return EXIT_FAILURE;
    }

    try {
        Graph g = readGraph(argv[1]);

        std::cout << "===== DYNAMIC GRAPH DSU =====\n"
                  << "Vertices: " << g.n << '\n'
                  << "Initial edges: " << g.initial.size() << '\n'
                  << "Inserted edges: " << g.updates.size() << '\n'
                  << "Connectivity: weakly connected components\n\n";

        CPU_DSU cpu(g.n);
        auto cpuStart = std::chrono::steady_clock::now();
        cpu.addEdges(g.initial);
        int beforeUpdates = cpu.count();
        cpu.addEdges(g.updates);
        int cpuComponents = cpu.count();
        auto cpuStop = std::chrono::steady_clock::now();

        double cpuMs =
            std::chrono::duration<double, std::milli>(cpuStop - cpuStart).count();
        std::vector<int> cpuLabels = cpu.labels();

        double gpuMs = 0.0;
        GPUResult gpu = runGPU(g, gpuMs);

        bool correct = (cpuLabels == gpu.labels) &&
                       (cpuComponents == gpu.components);

        std::cout << "===== RESULTS =====\n"
                  << "CPU components before updates: " << beforeUpdates << '\n'
                  << "CPU components after updates : " << cpuComponents << '\n'
                  << "GPU components after updates : " << gpu.components << '\n'
                  << "GPU edge-kernel passes       : " << gpu.passes << '\n'
                  << std::fixed << std::setprecision(4)
                  << "CPU time (ms): " << cpuMs << '\n'
                  << "GPU time (ms): " << gpuMs << '\n';

        if (gpuMs > 0.0) {
            std::cout << "CPU/GPU time ratio: " << cpuMs / gpuMs << "x\n";
        }

        std::cout << "Correctness: " << (correct ? "PASS" : "FAIL") << "\n\n"
                  << "Vertex : CPU label | GPU label\n";
        for (int v = 0; v < g.n; ++v) {
            std::cout << v << " : " << cpuLabels[v]
                      << " | " << gpu.labels[v] << '\n';
        }

        return correct ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
