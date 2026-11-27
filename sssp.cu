#include "common.cuh"
#include <cooperative_groups.h>
#include <algorithm>

namespace cg = cooperative_groups;
namespace {

constexpr unsigned long long DP_INF  = 0xFFFFFFFFFFFFFFFFull;
constexpr int SMALL_DEG = 32;    // degree < 32
constexpr int LARGE_DEG = 1024;  // degree >= 1024 -> one block per vertex 
constexpr int SHARED_Q  = 2048;  // shared-memory output queue entries per block

__device__ __forceinline__ unsigned long long pack_dp(dist_t d, vid_t p) {
    return (static_cast<unsigned long long>(d) << 32) |
           static_cast<unsigned long long>(static_cast<uint32_t>(p));
}
__device__ __forceinline__ dist_t dp_dist(unsigned long long x) {
    return static_cast<dist_t>(x >> 32);
}
__device__ __forceinline__ unsigned long long load_dp(const unsigned long long* p) {
    return *reinterpret_cast<const volatile unsigned long long*>(p);
}

// Relax arc u->v. Returns true if this thread strictly lowered dist[v].
__device__ __forceinline__ bool relax(unsigned long long* dp, vid_t u, dist_t du,
                                      vid_t v, weight_t w) {
    dist_t nd = du + w;
    unsigned long long cand = pack_dp(nd, u);
    if (cand >= load_dp(&dp[v])) return false;
    unsigned long long old = atomicMin(&dp[v], cand);
    return nd < dp_dist(old);
}

// Warp-aggregated push: active threads elect a leader that reserves space once.
__device__ __forceinline__ void push_aggregated(vid_t v, vid_t* q, int* q_size) {
    cg::coalesced_group g = cg::coalesced_threads();
    int base = 0;
    if (g.thread_rank() == 0) base = atomicAdd(q_size, static_cast<int>(g.size()));
    base = g.shfl(base, 0);
    q[base + g.thread_rank()] = v;
}

__device__ __forceinline__ void enqueue_global(vid_t v, unsigned* flag, vid_t* q, int* q_size) {
    if (atomicExch(&flag[v], 1u) == 0u) push_aggregated(v, q, q_size);
}

__global__ void k_init_dp(unsigned long long* dp, vid_t n, vid_t src) {
    for (vid_t v = blockIdx.x * blockDim.x + threadIdx.x; v < n; v += gridDim.x * blockDim.x)
        dp[v] = (v == src) ? pack_dp(0, src) : DP_INF;
}

// Thread-per-vertex relaxation with a shared-memory output queue.
__global__ void k_relax_thread(GraphView g, unsigned long long* dp,
                               const vid_t* __restrict__ fq, int fsize, bool clear_flag,
                               unsigned* flag, vid_t* next, int* next_size) {
    __shared__ vid_t sq[SHARED_Q];
    __shared__ int s_count, s_base;
    const int stride = gridDim.x * blockDim.x;
    for (int base = blockIdx.x * blockDim.x; base < fsize; base += stride) {
        if (threadIdx.x == 0) s_count = 0;
        __syncthreads();
        const int i = base + threadIdx.x;
        if (i < fsize) {
            const vid_t u = fq[i];
            if (clear_flag) { atomicExch(&flag[u], 0u); __threadfence(); }
            const dist_t du = dp_dist(load_dp(&dp[u]));
            if (du != DIST_INF) {
                const eid_t bdeg = g.base_deg(u), deg = g.degree(u);
                for (eid_t j = 0; j < deg; ++j) {
                    vid_t v; weight_t w; g.arc(u, j, bdeg, v, w);
                    if (relax(dp, u, du, v, w) && atomicExch(&flag[v], 1u) == 0u) {
                        int pos = atomicAdd(&s_count, 1);
                        if (pos < SHARED_Q) sq[pos] = v;
                        else push_aggregated(v, next, next_size);
                    }
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) { int c = min(s_count, SHARED_Q); s_base = c > 0 ? atomicAdd(next_size, c) : 0; }
        __syncthreads();
        const int c = min(s_count, SHARED_Q);
        for (int j = threadIdx.x; j < c; j += blockDim.x) next[s_base + j] = sq[j];
        __syncthreads();
    }
}

// Warp-per-vertex (medium degree): lanes stride over arcs of one vertex.
__global__ void k_relax_warp(GraphView g, unsigned long long* dp,
                              const vid_t* __restrict__ fq, int fsize,
                              unsigned* flag, vid_t* next, int* next_size) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int nwarps = (gridDim.x * blockDim.x) >> 5;
    for (int i = warp; i < fsize; i += nwarps) {
        const vid_t u = fq[i];
        const dist_t du = dp_dist(load_dp(&dp[u]));
        if (du == DIST_INF) continue;
        const eid_t bdeg = g.base_deg(u), deg = g.degree(u);
        for (eid_t j = lane; j < deg; j += 32) {
            vid_t v; weight_t w; g.arc(u, j, bdeg, v, w);
            if (relax(dp, u, du, v, w)) enqueue_global(v, flag, next, next_size);
        }
    }
}

// Block-per-vertex 
__global__ void k_relax_block(GraphView g, unsigned long long* dp,
                               const vid_t* __restrict__ fq, int fsize,
                               unsigned* flag, vid_t* next, int* next_size) {
    for (int i = blockIdx.x; i < fsize; i += gridDim.x) {
        const vid_t u = fq[i];
        const dist_t du = dp_dist(load_dp(&dp[u]));
        if (du == DIST_INF) continue;
        const eid_t bdeg = g.base_deg(u), deg = g.degree(u);
        for (eid_t j = threadIdx.x; j < deg; j += blockDim.x) {
            vid_t v; weight_t w; g.arc(u, j, bdeg, v, w);
            if (relax(dp, u, du, v, w)) enqueue_global(v, flag, next, next_size);
        }
    }
}

// Split frontier into degree bins and clear frontier flags.
__global__ void k_classify(GraphView g, const vid_t* __restrict__ fq, int fsize,
                            unsigned* flag, vid_t* qs, vid_t* qm, vid_t* ql, int* counters) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < fsize; i += gridDim.x * blockDim.x) {
        const vid_t u = fq[i];
        atomicExch(&flag[u], 0u);
        const eid_t d = g.degree(u);
        if      (d < SMALL_DEG) push_aggregated(u, qs, &counters[1]);
        else if (d < LARGE_DEG) push_aggregated(u, qm, &counters[2]);
        else                    push_aggregated(u, ql, &counters[3]);
    }
}

// Topology-driven Bellman-Ford: one kernel call = one iteration over all vertices.
__global__ void k_bellman_ford(GraphView g, unsigned long long* dp, int* changed) {
    int any = 0;
    for (vid_t u = blockIdx.x * blockDim.x + threadIdx.x; u < g.n; u += gridDim.x * blockDim.x) {
        const dist_t du = dp_dist(load_dp(&dp[u]));
        if (du == DIST_INF) continue;
        const eid_t bdeg = g.base_deg(u), deg = g.degree(u);
        for (eid_t j = 0; j < deg; ++j) {
            vid_t v; weight_t w; g.arc(u, j, bdeg, v, w);
            if (relax(dp, u, du, v, w)) any = 1;
        }
    }
    if (any) *changed = 1;
}

// Dynamic SSSP seed: test every inserted arc; improved destinations become the first frontier.
__global__ void k_seed_insertions(unsigned long long* dp, const vid_t* __restrict__ eu,
                                  const vid_t* __restrict__ ev, const weight_t* __restrict__ ew,
                                  int k, bool directed, unsigned* flag, vid_t* q, int* q_size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < k; i += gridDim.x * blockDim.x) {
        const vid_t a = eu[i], b = ev[i]; const weight_t w = ew[i];
        dist_t da = dp_dist(load_dp(&dp[a]));
        if (da != DIST_INF && relax(dp, a, da, b, w)) enqueue_global(b, flag, q, q_size);
        if (!directed) {
            dist_t db = dp_dist(load_dp(&dp[b]));
            if (db != DIST_INF && relax(dp, b, db, a, w)) enqueue_global(a, flag, q, q_size);
        }
    }
}

}  
void GpuSSSP::init(vid_t n, int block_size, bool load_balance) {
    release();
    n_ = n; block_ = block_size; lb_ = load_balance;
    d_dp_       = dmalloc<unsigned long long>(n_);
    d_flag_     = dmalloc<unsigned>(n_);
    d_q_[0]     = dmalloc<vid_t>(n_);
    d_q_[1]     = dmalloc<vid_t>(n_);
    d_qs_       = dmalloc<vid_t>(n_);
    d_qm_       = dmalloc<vid_t>(n_);
    d_ql_       = dmalloc<vid_t>(n_);
    d_counters_ = dmalloc<int>(8);
    CUDA_CHECK(cudaMemset(d_flag_,     0, n_ * sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(d_counters_, 0, 8  * sizeof(int)));
    k_init_dp<<<grid_for(n_, 256), 256>>>(d_dp_, n_, -1);  // all unreachable
    CUDA_CHECK_LAST();
}

void GpuSSSP::release() {
    dfree(d_dp_,       n_);
    dfree(d_flag_,     n_);
    dfree(d_q_[0],     n_);
    dfree(d_q_[1],     n_);
    dfree(d_qs_,       n_);
    dfree(d_qm_,       n_);
    dfree(d_ql_,       n_);
    dfree(d_counters_, 8);
}

void GpuSSSP::reset_source(vid_t src) {
    k_init_dp<<<grid_for(n_, block_), block_>>>(d_dp_, n_, src);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemset(d_flag_, 0, n_ * sizeof(unsigned)));
}

void GpuSSSP::propagate(const GraphView& g, int fsize, SSSPStats& st) {
    while (fsize > 0) {
        vid_t* fq   = d_q_[cur_];
        vid_t* next = d_q_[cur_ ^ 1];
        CUDA_CHECK(cudaMemset(d_counters_, 0, 4 * sizeof(int)));

        if (lb_) {
            k_classify<<<grid_for(fsize, block_), block_>>>(g, fq, fsize, d_flag_,
                                                             d_qs_, d_qm_, d_ql_, d_counters_);
            CUDA_CHECK_LAST();
            int bins[4];
            CUDA_CHECK(cudaMemcpy(bins, d_counters_, 4 * sizeof(int), cudaMemcpyDeviceToHost));
            if (bins[1] > 0)
                k_relax_thread<<<grid_for(bins[1], block_), block_>>>(
                    g, d_dp_, d_qs_, bins[1], false, d_flag_, next, d_counters_);
            if (bins[2] > 0)
                k_relax_warp<<<grid_for(static_cast<long long>(bins[2]) * 32, block_), block_>>>(
                    g, d_dp_, d_qm_, bins[2], d_flag_, next, d_counters_);
            if (bins[3] > 0)
                k_relax_block<<<std::min(bins[3], 65535), block_>>>(
                    g, d_dp_, d_ql_, bins[3], d_flag_, next, d_counters_);
        } else {
            k_relax_thread<<<grid_for(fsize, block_), block_>>>(
                g, d_dp_, fq, fsize, true, d_flag_, next, d_counters_);
        }
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpy(&fsize, d_counters_, sizeof(int), cudaMemcpyDeviceToHost));
        cur_ ^= 1;
        st.iterations++;
        st.frontier_pushes += fsize;
    }
}

SSSPStats GpuSSSP::run_bellman_ford(const GraphView& g, vid_t src, int max_iters) {
    SSSPStats st;
    GpuTimer total; total.start();
    reset_source(src);
    if (max_iters < 0) max_iters = n_;
    int changed = 1;
    while (changed && st.iterations < max_iters) {
        CUDA_CHECK(cudaMemset(d_counters_ + 4, 0, sizeof(int)));
        k_bellman_ford<<<grid_for(n_, block_), block_>>>(g, d_dp_, d_counters_ + 4);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpy(&changed, d_counters_ + 4, sizeof(int), cudaMemcpyDeviceToHost));
        st.iterations++;
    }
    st.total_ms = st.propagate_ms = total.stop_ms();
    return st;
}

SSSPStats GpuSSSP::run_frontier(const GraphView& g, vid_t src) {
    SSSPStats st;
    GpuTimer total; total.start();
    reset_source(src);
    const unsigned one = 1;
    CUDA_CHECK(cudaMemcpy(d_q_[cur_],     &src, sizeof(vid_t),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_flag_ + src,  &one, sizeof(unsigned), cudaMemcpyHostToDevice));
    st.frontier_pushes = 1;
    propagate(g, 1, st);
    st.total_ms = st.propagate_ms = total.stop_ms();
    return st;
}

SSSPStats GpuSSSP::update_insertions(const GraphView& g, const vid_t* d_u, const vid_t* d_v,
                                     const weight_t* d_w, int k, bool directed) {
    SSSPStats st;
    if (k == 0) return st;
    GpuTimer total, phase;
    total.start(); phase.start();
    CUDA_CHECK(cudaMemset(d_counters_, 0, sizeof(int)));
    k_seed_insertions<<<grid_for(k, block_), block_>>>(d_dp_, d_u, d_v, d_w, k, directed,
                                                        d_flag_, d_q_[cur_], d_counters_);
    CUDA_CHECK_LAST();
    int fsize = 0;
    CUDA_CHECK(cudaMemcpy(&fsize, d_counters_, sizeof(int), cudaMemcpyDeviceToHost));
    st.seed_ms = phase.stop_ms();
    st.seeds   = fsize;
    st.frontier_pushes = fsize;

    phase.start();
    propagate(g, fsize, st);
    st.propagate_ms = phase.stop_ms();
    st.total_ms     = total.stop_ms();
    return st;
}

void GpuSSSP::distances(std::vector<dist_t>& dist, std::vector<vid_t>* pred) const {
    std::vector<unsigned long long> h(n_);
    CUDA_CHECK(cudaMemcpy(h.data(), d_dp_, n_ * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    dist.resize(n_);
    if (pred) pred->resize(n_);
    for (vid_t v = 0; v < n_; ++v) {
        dist[v] = static_cast<dist_t>(h[v] >> 32);
        if (pred) (*pred)[v] = static_cast<vid_t>(h[v] & 0xFFFFFFFFull);
    }
}
