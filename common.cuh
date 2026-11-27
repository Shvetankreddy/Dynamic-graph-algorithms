#pragma once
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

using vid_t    = int32_t;
using eid_t    = int32_t;
using weight_t = uint32_t;
using dist_t   = uint32_t;


static constexpr dist_t DIST_INF = 0xFFFFFFFFu;

// One directed or undirected edge.
struct Edge {
    vid_t u;
    vid_t v;
    weight_t w;
};

//timer for host-side 
class HostTimer {
public:
    void start() { t0_ = std::chrono::high_resolution_clock::now(); }
    double stop_ms() const {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point t0_;
};

inline void fail(const std::string& msg) {
    std::fprintf(stderr, "ERROR: %s\n", msg.c_str());
    std::exit(EXIT_FAILURE);
}

// CUDA error checking, timers, and tracked allocator 
#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err__ = (call);                                                       \
        if (err__ != cudaSuccess) {                                                       \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,        \
                         __LINE__, cudaGetErrorString(err__));                            \
            std::exit(EXIT_FAILURE);                                                      \
        }                                                                                 \
    } while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

// GPU event timer 
class GpuTimer {
public:
    GpuTimer()  { CUDA_CHECK(cudaEventCreate(&start_)); CUDA_CHECK(cudaEventCreate(&stop_)); }
    ~GpuTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void  start()     { CUDA_CHECK(cudaEventRecord(start_)); }
    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
private:
    cudaEvent_t start_, stop_;
};
struct DeviceMemStats { size_t current = 0, peak = 0; };
DeviceMemStats& device_mem_stats();   // defined in dsu.cu

template <typename T>
T* dmalloc(size_t count) {
    T* p = nullptr;
    if (count == 0) count = 1;
    CUDA_CHECK(cudaMalloc(&p, count * sizeof(T)));
    auto& s = device_mem_stats();
    s.current += count * sizeof(T);
    if (s.current > s.peak) s.peak = s.current;
    return p;
}
template <typename T>
void dfree(T*& p, size_t count) {
    if (!p) return;
    if (count == 0) count = 1;
    CUDA_CHECK(cudaFree(p));
    device_mem_stats().current -= count * sizeof(T);
    p = nullptr;
}

// Defined in dsu.cu.
int grid_for(long long work, int block);

// COO edge list. For an undirected graph each edge is stored once;
struct EdgeList {
    vid_t n = 0;
    bool directed = false;
    std::vector<Edge> edges;
};
// Compressed Sparse Row (out-arcs).
// row_ptr has n+1 entries; arcs of u are col[row_ptr[u] .. row_ptr[u+1]).
struct CSR {
    vid_t n = 0;
    eid_t m = 0;
    std::vector<eid_t> row_ptr;
    std::vector<vid_t> col;
    std::vector<weight_t> w;
};

// Build a CSR from an edge list. Undirected edges contribute two arcs each.
CSR build_csr(vid_t n, const std::vector<Edge>& edges, bool directed);

struct HostArc { vid_t v; weight_t w; };

class HostDynGraph {
public:
    HostDynGraph() = default;
    HostDynGraph(const EdgeList& initial);

    void insert_edges(const std::vector<Edge>& batch);

    vid_t  n()         const { return base_.n; }
    bool   directed()  const { return directed_; }
    size_t num_arcs()  const { return static_cast<size_t>(base_.m) + m_delta_; }
    size_t num_edges() const { return directed_ ? num_arcs() : num_arcs() / 2; }
    const CSR& base()  const { return base_; }

    template <typename F>
    void for_each_out(vid_t u, F&& f) const {
        for (eid_t e = base_.row_ptr[u]; e < base_.row_ptr[u + 1]; ++e)
            f(base_.col[e], base_.w[e]);
        for (const HostArc& a : delta_[u]) f(a.v, a.w);
    }

    size_t degree(vid_t u) const {
        return static_cast<size_t>(base_.row_ptr[u+1] - base_.row_ptr[u]) + delta_[u].size();
    }

    bool has_arc(vid_t u, vid_t v, weight_t w) const;

private:
    bool directed_ = false;
    CSR base_;
    std::vector<std::vector<HostArc>> delta_;
    size_t m_delta_ = 0;
};

// Return the vertex with the largest out-degree (used as default SSSP source).
vid_t max_degree_vertex(const CSR& g);
// Graph generators (all weights uniform in [1, max_w])
EdgeList gen_uniform(vid_t n, size_t m, weight_t max_w, uint64_t seed, bool directed);
EdgeList gen_rmat(int scale, int edge_factor, weight_t max_w, uint64_t seed, bool directed,
                  double a = 0.57, double b = 0.19, double c = 0.19);
EdgeList gen_grid(int rows, int cols, weight_t max_w, uint64_t seed, bool directed);

EdgeList load_snap(const std::string& path, weight_t max_w, uint64_t seed, bool directed);

EdgeList load_dimacs(const std::string& path, bool directed);

// Split an edge list into the initial graph and an insertion stream.
// Edges are shuffled; the first init_frac fraction is the initial graph.
void split_initial_and_stream(const EdgeList& full, double init_frac, uint64_t seed,EdgeList& initial, std::vector<Edge>& stream);

// Lightweight view passed by value to CUDA kernels.
struct GraphView {
    vid_t n;
    const eid_t* __restrict__ row;
    const vid_t* __restrict__ col;
    const weight_t* __restrict__ w;
    const eid_t* __restrict__ drow;
    const vid_t* __restrict__ dcol;
    const weight_t* __restrict__ dw;

    __device__ __forceinline__ eid_t base_begin(vid_t u)  const { return row[u]; }
    __device__ __forceinline__ eid_t base_deg(vid_t u)    const { return row[u+1] - row[u]; }
    __device__ __forceinline__ eid_t delta_begin(vid_t u) const { return drow[u]; }
    __device__ __forceinline__ eid_t delta_deg(vid_t u)   const { return drow[u+1] - drow[u]; }
    __device__ __forceinline__ eid_t degree(vid_t u)      const { return base_deg(u) + delta_deg(u); }

    // i-th out-arc of u across base and delta (0 <= i < degree(u)).
    __device__ __forceinline__ void arc(vid_t u, eid_t i, eid_t bdeg,vid_t& v, weight_t& wt) const {
        if (i < bdeg) {
            eid_t e = row[u] + i;  v = col[e]; wt = w[e];
        } else {
            eid_t e = drow[u] + (i - bdeg); v = dcol[e]; wt = dw[e];
        }
    }
};

class DeviceDynGraph {
public:
    DeviceDynGraph() = default;
    ~DeviceDynGraph() { release(); }
    DeviceDynGraph(const DeviceDynGraph&) = delete;
    DeviceDynGraph& operator=(const DeviceDynGraph&) = delete;

    void init(const CSR& base, bool directed, size_t max_batch_edges, double merge_frac);
    void release();

    float upload_batch(const std::vector<Edge>& batch);
    float apply_batch(bool* merged);

    GraphView view() const {
        return GraphView{n_, d_row_, d_col_, d_w_, d_drow_, d_dcol_, d_dw_};
    }

    vid_t  n()            const { return n_; }
    bool   directed()     const { return directed_; }
    eid_t  m_base()       const { return m_base_; }
    eid_t  m_delta()      const { return m_delta_; }
    eid_t  m_total()      const { return m_base_ + m_delta_; }
    int    staged_edges() const { return staged_; }

    const vid_t*    staged_u() const { return d_es_; }
    const vid_t*    staged_v() const { return d_ed_; }
    const weight_t* staged_w() const { return d_ew_; }

private:
    void merge_delta_into_base();
    void rebuild_delta_csr();
    void grow_delta(size_t need);

    vid_t  n_ = 0;
    bool   directed_   = false;
    double merge_frac_ = 0.1;

    eid_t    m_base_  = 0;
    eid_t*   d_row_   = nullptr;
    vid_t*   d_col_   = nullptr;
    weight_t* d_w_    = nullptr;

    eid_t    m_delta_  = 0;
    size_t   cap_delta_ = 0;
    eid_t*   d_drow_  = nullptr;
    vid_t*   d_dsrc_  = nullptr;
    vid_t*   d_dcol_  = nullptr;
    weight_t* d_dw_   = nullptr;

    size_t  cap_batch_   = 0;
    int     staged_      = 0;
    vid_t*  d_es_        = nullptr;
    vid_t*  d_ed_        = nullptr;
    weight_t* d_ew_      = nullptr;
    Edge*   h_stage_     = nullptr;   
    Edge*   d_stage_aos_ = nullptr;
};


//GPU Disjoint Set Union  
class GpuDSU {
public:
    GpuDSU() = default;
    ~GpuDSU() { release(); }
    GpuDSU(const GpuDSU&) = delete;
    GpuDSU& operator=(const GpuDSU&) = delete;

    void init(vid_t n, size_t max_queries);
    void release();
    void reset();   // set every vertex as its own root

    // Union every edge of a device edge array; returns kernel time (ms) incl.
    float union_edges(const vid_t* d_u, const vid_t* d_v, int k);
    // Static recompute: reset and union every arc of the current graph.
    float recompute_from_graph(const GraphView& g, bool directed);
    // Answer "are a and b connected?" for a batch of pairs; returns kernel time (ms).
    float query(const std::vector<vid_t>& qa, const std::vector<vid_t>& qb,
                std::vector<uint8_t>& out);

    vid_t components() const;                   
    void  roots(std::vector<vid_t>& out) const; 

private:
    void compress();

    vid_t  n_ = 0;
    unsigned long long* d_word_       = nullptr;
    int*                d_components_ = nullptr;
    size_t cap_q_ = 0;
    vid_t*   d_qa_   = nullptr;
    vid_t*   d_qb_   = nullptr;
    uint8_t* d_qout_ = nullptr;
};

// GPU SSSP  
struct SSSPStats {
    float seed_ms      = 0.f;
    float propagate_ms = 0.f;
    float total_ms     = 0.f;
    int   iterations   = 0;
    long long frontier_pushes = 0;
    int   seeds        = 0;
};

class GpuSSSP {
public:
    GpuSSSP() = default;
    ~GpuSSSP() { release(); }
    GpuSSSP(const GpuSSSP&) = delete;
    GpuSSSP& operator=(const GpuSSSP&) = delete;

    void init(vid_t n, int block_size, bool load_balance);
    void release();

    SSSPStats run_bellman_ford(const GraphView& g, vid_t src, int max_iters = -1);
    SSSPStats run_frontier(const GraphView& g, vid_t src);
    // Graph must already contain the batch. d_u/d_v/d_w are staged batch edges (device ptrs).
    SSSPStats update_insertions(const GraphView& g, const vid_t* d_u, const vid_t* d_v,
                                const weight_t* d_w, int k, bool directed);

    void distances(std::vector<dist_t>& dist, std::vector<vid_t>* pred = nullptr) const;
    void set_load_balance(bool lb) { lb_ = lb; }

private:
    void reset_source(vid_t src);
    void propagate(const GraphView& g, int fsize, SSSPStats& st);

    vid_t n_    = 0;
    int   block_ = 256;
    bool  lb_   = true;
    int   cur_  = 0;

    unsigned long long* d_dp_       = nullptr;
    unsigned*           d_flag_     = nullptr;
    vid_t*              d_q_[2]     = {nullptr, nullptr};
    vid_t*              d_qs_       = nullptr;
    vid_t*              d_qm_       = nullptr;
    vid_t*              d_ql_       = nullptr;
    int*                d_counters_ = nullptr;
};

// Sequential DSU with path compression and union by rank.
class CpuDSU {
public:
    explicit CpuDSU(vid_t n = 0) { reset(n); }
    void   reset(vid_t n);
    vid_t  find(vid_t x);
    bool   unite(vid_t a, vid_t b);   // true if a merge happened
    vid_t  components() const { return components_; }
private:
    std::vector<vid_t>   parent_;
    std::vector<uint8_t> rank_;
    vid_t components_ = 0;
};

// Recompute connected components from scratch over the whole current graph.
vid_t cpu_cc_recompute(const HostDynGraph& g);

// Dijkstra (binary heap) from src over the current graph. dist[v]=DIST_INF if unreachable.
void cpu_dijkstra(const HostDynGraph& g, vid_t src, std::vector<dist_t>& dist);

// Incremental SSSP on the CPU: dist holds correct distances BEFORE the batch;
// g already contains the batch. Returns the number of vertices whose distance decreased.
size_t cpu_sssp_incremental(const HostDynGraph& g, const std::vector<Edge>& batch,
                             std::vector<dist_t>& dist);
