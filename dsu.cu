#include "common.cuh"
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <cstring>
using namespace std;

DeviceMemStats& device_mem_stats() {
    static DeviceMemStats s;
    return s;
}

int grid_for(long long work, int block) {
    static int max_blocks = 0;
    if (max_blocks == 0) {
        int dev = 0, sms = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        max_blocks = sms * 32;
    }
    long long g = (work + block - 1) / block;
    if (g < 1) g = 1;
    return static_cast<int>(min<long long>(g, max_blocks));
}


//DeviceDynGraph kernels and implementation
// Unpack an AoS batch into SoA arrays 
__global__ void k_unpack_batch(const Edge* __restrict__ in, int k,
                                vid_t* __restrict__ u, vid_t* __restrict__ v,
                                weight_t* __restrict__ w) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < k; i += gridDim.x * blockDim.x) {
        Edge e = in[i];
        u[i] = e.u; v[i] = e.v; w[i] = e.w;
    }
}

// Append batch arcs to the end of the delta COO (both directions for undirected graphs).
__global__ void k_append_arcs(const vid_t* __restrict__ eu, const vid_t* __restrict__ ev,
                               const weight_t* __restrict__ ew, int k, bool directed,
                               vid_t* __restrict__ dsrc, vid_t* __restrict__ dcol,
                               weight_t* __restrict__ dw, eid_t offset) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < k; i += gridDim.x * blockDim.x) {
        vid_t a = eu[i], b = ev[i]; weight_t wt = ew[i];
        dsrc[offset + i] = a; dcol[offset + i] = b; dw[offset + i] = wt;
        if (!directed) {
            dsrc[offset + k + i] = b; dcol[offset + k + i] = a; dw[offset + k + i] = wt;
        }
    }
}

// Expand CSR row offsets into an explicit source array (CSR -> COO conversion).
__global__ void k_expand_rows(const eid_t* __restrict__ row, vid_t n, vid_t* __restrict__ src) {
    for (vid_t u = blockIdx.x * blockDim.x + threadIdx.x; u < n; u += gridDim.x * blockDim.x)
        for (eid_t e = row[u]; e < row[u + 1]; ++e) src[e] = u;
}

// Sort COO (src, col, w) by source vertex and write CSR row offsets.
static void coo_to_csr_inplace(vid_t* src, vid_t* col, weight_t* w, eid_t m, vid_t n, eid_t* row) {
    if (m > 0) {
        auto vals = thrust::make_zip_iterator(thrust::make_tuple(col, w));
        thrust::sort_by_key(thrust::device, src, src + m, vals);
    }
    thrust::lower_bound(thrust::device, src, src + m,
                        thrust::counting_iterator<vid_t>(0),
                        thrust::counting_iterator<vid_t>(n + 1), row);
}

void DeviceDynGraph::init(const CSR& base, bool directed, size_t max_batch_edges,
                          double merge_frac) {
    release();
    n_ = base.n; directed_ = directed; merge_frac_ = merge_frac; m_base_ = base.m;

    d_row_ = dmalloc<eid_t>(static_cast<size_t>(n_) + 1);
    d_col_ = dmalloc<vid_t>(m_base_);
    d_w_   = dmalloc<weight_t>(m_base_);
    CUDA_CHECK(cudaMemcpy(d_row_, base.row_ptr.data(), (n_ + 1) * sizeof(eid_t), cudaMemcpyHostToDevice));
    if (m_base_ > 0) {
        CUDA_CHECK(cudaMemcpy(d_col_, base.col.data(), m_base_ * sizeof(vid_t),    cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w_,   base.w.data(),   m_base_ * sizeof(weight_t), cudaMemcpyHostToDevice));
    }

    d_drow_ = dmalloc<eid_t>(static_cast<size_t>(n_) + 1);
    CUDA_CHECK(cudaMemset(d_drow_, 0, (n_ + 1) * sizeof(eid_t)));
    m_delta_ = 0;
    grow_delta(max<size_t>(1024, 2 * max_batch_edges));

    cap_batch_ = max<size_t>(1, max_batch_edges);
    d_es_ = dmalloc<vid_t>(cap_batch_);
    d_ed_ = dmalloc<vid_t>(cap_batch_);
    d_ew_ = dmalloc<weight_t>(cap_batch_);
    d_stage_aos_ = dmalloc<Edge>(cap_batch_);
    CUDA_CHECK(cudaMallocHost(&h_stage_, cap_batch_ * sizeof(Edge)));
}

void DeviceDynGraph::release() {
    dfree(d_row_,       static_cast<size_t>(n_) + 1);
    dfree(d_col_,       m_base_);
    dfree(d_w_,         m_base_);
    dfree(d_drow_,      static_cast<size_t>(n_) + 1);
    dfree(d_dsrc_,      cap_delta_);
    dfree(d_dcol_,      cap_delta_);
    dfree(d_dw_,        cap_delta_);
    dfree(d_es_,        cap_batch_);
    dfree(d_ed_,        cap_batch_);
    dfree(d_ew_,        cap_batch_);
    dfree(d_stage_aos_, cap_batch_);
    if (h_stage_) { cudaFreeHost(h_stage_); h_stage_ = nullptr; }
    cap_delta_ = 0; cap_batch_ = 0; m_base_ = 0; m_delta_ = 0;
}

void DeviceDynGraph::grow_delta(size_t need) {
    if (need <= cap_delta_) return;
    size_t cap = max(need, 2 * cap_delta_);
    vid_t*    nsrc = dmalloc<vid_t>(cap);
    vid_t*    ncol = dmalloc<vid_t>(cap);
    weight_t* nw   = dmalloc<weight_t>(cap);
    if (m_delta_ > 0) {
        CUDA_CHECK(cudaMemcpy(nsrc, d_dsrc_, m_delta_ * sizeof(vid_t),    cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(ncol, d_dcol_, m_delta_ * sizeof(vid_t),    cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(nw,   d_dw_,   m_delta_ * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    }
    dfree(d_dsrc_, cap_delta_); dfree(d_dcol_, cap_delta_); dfree(d_dw_, cap_delta_);
    d_dsrc_ = nsrc; d_dcol_ = ncol; d_dw_ = nw; cap_delta_ = cap;
}

float DeviceDynGraph::upload_batch(const vector<Edge>& batch) {
    if (batch.size() > cap_batch_) fail("batch larger than staging buffer");
    HostTimer t; t.start();
    staged_ = static_cast<int>(batch.size());
    if (staged_ > 0) {
        memcpy(h_stage_, batch.data(), batch.size() * sizeof(Edge));
        CUDA_CHECK(cudaMemcpy(d_stage_aos_, h_stage_, batch.size() * sizeof(Edge), cudaMemcpyHostToDevice));
        k_unpack_batch<<<grid_for(staged_, 256), 256>>>(d_stage_aos_, staged_, d_es_, d_ed_, d_ew_);
        CUDA_CHECK_LAST();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return static_cast<float>(t.stop_ms());
}

void DeviceDynGraph::rebuild_delta_csr() {
    coo_to_csr_inplace(d_dsrc_, d_dcol_, d_dw_, m_delta_, n_, d_drow_);
    CUDA_CHECK_LAST();
}

void DeviceDynGraph::merge_delta_into_base() {
    const eid_t total = m_base_ + m_delta_;
    vid_t*    src = dmalloc<vid_t>(total);
    vid_t*    col = dmalloc<vid_t>(total);
    weight_t* w   = dmalloc<weight_t>(total);

    if (m_base_ > 0) {
        k_expand_rows<<<grid_for(n_, 256), 256>>>(d_row_, n_, src);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaMemcpy(col, d_col_, m_base_ * sizeof(vid_t),    cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(w,   d_w_,   m_base_ * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    }
    if (m_delta_ > 0) {
        CUDA_CHECK(cudaMemcpy(src + m_base_, d_dsrc_, m_delta_ * sizeof(vid_t),    cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(col + m_base_, d_dcol_, m_delta_ * sizeof(vid_t),    cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(w   + m_base_, d_dw_,   m_delta_ * sizeof(weight_t), cudaMemcpyDeviceToDevice));
    }
    coo_to_csr_inplace(src, col, w, total, n_, d_row_);
    CUDA_CHECK_LAST();

    dfree(d_col_, m_base_); dfree(d_w_, m_base_); dfree(src, total);
    d_col_ = col; d_w_ = w; m_base_ = total;
    m_delta_ = 0;
    CUDA_CHECK(cudaMemset(d_drow_, 0, (n_ + 1) * sizeof(eid_t)));
}

float DeviceDynGraph::apply_batch(bool* merged) {
    GpuTimer t; t.start();
    *merged = false;
    if (staged_ > 0) {
        const eid_t arcs = directed_ ? staged_ : 2 * staged_;
        grow_delta(static_cast<size_t>(m_delta_) + arcs);
        k_append_arcs<<<grid_for(staged_, 256), 256>>>(d_es_, d_ed_, d_ew_, staged_, directed_,
                                                        d_dsrc_, d_dcol_, d_dw_, m_delta_);
        CUDA_CHECK_LAST();
        m_delta_ += arcs;

        const double threshold = merge_frac_ * max<double>(m_base_, 1024.0);
        if (merge_frac_ > 0.0 && m_delta_ > threshold) {
            merge_delta_into_base();
            *merged = true;
        } else {
            rebuild_delta_csr();
        }
    }
    return t.stop_ms();
}
//GpuDSU kernels and implementation

namespace {

__device__ __forceinline__ vid_t w_parent(unsigned long long w) {
    return static_cast<vid_t>(w & 0xFFFFFFFFull);
}
__device__ __forceinline__ unsigned w_rank(unsigned long long w) {
    return static_cast<unsigned>(w >> 32);
}
__device__ __forceinline__ unsigned long long pack(unsigned rank, vid_t parent) {
    return (static_cast<unsigned long long>(rank) << 32) |
           static_cast<unsigned long long>(static_cast<uint32_t>(parent));
}
__device__ __forceinline__ unsigned long long load_word(const unsigned long long* p) {
    return *reinterpret_cast<const volatile unsigned long long*>(p);
}

// Find with path halving (concurrent unions may move roots; callers re-validate with CAS).
__device__ vid_t dsu_find(unsigned long long* word, vid_t x) {
    while (true) {
        unsigned long long wx = load_word(&word[x]);
        vid_t p = w_parent(wx);
        if (p == x) return x;
        unsigned long long wp = load_word(&word[p]);
        vid_t gp = w_parent(wp);
        if (gp == p) return p;
        // Point x at its grandparent (keeps x's frozen rank); failure is harmless.
        atomicCAS(&word[x], wx, pack(w_rank(wx), gp));
        x = gp;
    }
}

// Returns true if this call merged two different sets.
__device__ bool dsu_union(unsigned long long* word, vid_t a, vid_t b) {
    while (true) {
        vid_t ra = dsu_find(word, a);
        vid_t rb = dsu_find(word, b);
        if (ra == rb) return false;

        unsigned long long wa = load_word(&word[ra]);
        unsigned long long wb = load_word(&word[rb]);
        if (w_parent(wa) != ra || w_parent(wb) != rb) continue;   // root moved, retry

        unsigned rka = w_rank(wa), rkb = w_rank(wb);
        // Child = smaller (rank, id) key; parent = larger key.
        bool a_is_child = (rka < rkb) || (rka == rkb && ra < rb);
        vid_t child  = a_is_child ? ra : rb;
        vid_t parent = a_is_child ? rb : ra;
        unsigned long long wchild = a_is_child ? wa : wb;
        unsigned rk_child  = a_is_child ? rka : rkb;
        unsigned rk_parent = a_is_child ? rkb : rka;

        if (atomicCAS(&word[child], wchild, pack(rk_child, parent)) == wchild) {
            if (rk_child == rk_parent) {
                // Equal ranks: bump the new parent's rank while it is still a root.
                unsigned long long wp = load_word(&word[parent]);
                while (w_parent(wp) == parent && w_rank(wp) == rk_parent) {
                    unsigned long long old = atomicCAS(&word[parent], wp, pack(rk_parent + 1, parent));
                    if (old == wp) break;
                    wp = old;
                }
            }
            return true;
        }
        // CAS lost a race; retry from the top.
    }
}

__global__ void k_dsu_init(unsigned long long* word, vid_t n) {
    for (vid_t v = blockIdx.x * blockDim.x + threadIdx.x; v < n; v += gridDim.x * blockDim.x)
        word[v] = pack(0u, v);
}

__global__ void k_dsu_union_edges(unsigned long long* word, const vid_t* __restrict__ eu,
                                  const vid_t* __restrict__ ev, int k, int* components) {
    int merged = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < k; i += gridDim.x * blockDim.x)
        merged += dsu_union(word, eu[i], ev[i]) ? 1 : 0;
    // Warp reduction so only one atomic per warp hits the global counter.
    for (int off = 16; off > 0; off >>= 1) merged += __shfl_down_sync(0xFFFFFFFFu, merged, off);
    if ((threadIdx.x & 31) == 0 && merged) atomicSub(components, merged);
}

// Static recompute: one thread per vertex walks all its arcs (base + delta).
__global__ void k_dsu_union_graph(unsigned long long* word, GraphView g, bool directed,
                                  int* components) {
    int merged = 0;
    for (vid_t u = blockIdx.x * blockDim.x + threadIdx.x; u < g.n; u += gridDim.x * blockDim.x) {
        eid_t bdeg = g.base_deg(u), deg = g.degree(u);
        for (eid_t i = 0; i < deg; ++i) {
            vid_t v; weight_t wt;
            g.arc(u, i, bdeg, v, wt);
            if (directed || u < v) merged += dsu_union(word, u, v) ? 1 : 0;
        }
    }
    for (int off = 16; off > 0; off >>= 1) merged += __shfl_down_sync(0xFFFFFFFFu, merged, off);
    if ((threadIdx.x & 31) == 0 && merged) atomicSub(components, merged);
}

// Full path compression (safe to use plain stores because no concurrent unions run here).
__global__ void k_dsu_compress(unsigned long long* word, vid_t n) {
    for (vid_t v = blockIdx.x * blockDim.x + threadIdx.x; v < n; v += gridDim.x * blockDim.x) {
        vid_t r = v;
        while (true) { vid_t p = w_parent(load_word(&word[r])); if (p == r) break; r = p; }
        unsigned long long wv = load_word(&word[v]);
        if (w_parent(wv) != r) atomicCAS(&word[v], wv, pack(w_rank(wv), r));
    }
}

__global__ void k_dsu_query(unsigned long long* word, const vid_t* __restrict__ qa,
                            const vid_t* __restrict__ qb, int k, uint8_t* __restrict__ out) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < k; i += gridDim.x * blockDim.x)
        out[i] = dsu_find(word, qa[i]) == dsu_find(word, qb[i]) ? 1 : 0;
}

__global__ void k_dsu_roots(unsigned long long* word, vid_t n, vid_t* out) {
    for (vid_t v = blockIdx.x * blockDim.x + threadIdx.x; v < n; v += gridDim.x * blockDim.x)
        out[v] = dsu_find(word, v);
}

}

void GpuDSU::init(vid_t n, size_t max_queries) {
    release();
    n_ = n;
    d_word_       = dmalloc<unsigned long long>(n_);
    d_components_ = dmalloc<int>(1);
    cap_q_ = max<size_t>(1, max_queries);
    d_qa_   = dmalloc<vid_t>(cap_q_);
    d_qb_   = dmalloc<vid_t>(cap_q_);
    d_qout_ = dmalloc<uint8_t>(cap_q_);
    reset();
}

void GpuDSU::release() {
    dfree(d_word_,       n_);
    dfree(d_components_, 1);
    dfree(d_qa_,   cap_q_);
    dfree(d_qb_,   cap_q_);
    dfree(d_qout_, cap_q_);
    cap_q_ = 0;
}

void GpuDSU::reset() {
    k_dsu_init<<<grid_for(n_, 256), 256>>>(d_word_, n_);
    CUDA_CHECK_LAST();
    int c = n_;
    CUDA_CHECK(cudaMemcpy(d_components_, &c, sizeof(int), cudaMemcpyHostToDevice));
}

void GpuDSU::compress() {
    k_dsu_compress<<<grid_for(n_, 256), 256>>>(d_word_, n_);
    CUDA_CHECK_LAST();
}

float GpuDSU::union_edges(const vid_t* d_u, const vid_t* d_v, int k) {
    GpuTimer t; t.start();
    if (k > 0) {
        k_dsu_union_edges<<<grid_for(k, 256), 256>>>(d_word_, d_u, d_v, k, d_components_);
        CUDA_CHECK_LAST();
        compress();
    }
    return t.stop_ms();
}

float GpuDSU::recompute_from_graph(const GraphView& g, bool directed) {
    GpuTimer t; t.start();
    reset();
    k_dsu_union_graph<<<grid_for(n_, 256), 256>>>(d_word_, g, directed, d_components_);
    CUDA_CHECK_LAST();
    compress();
    return t.stop_ms();
}

float GpuDSU::query(const vector<vid_t>& qa, const vector<vid_t>& qb,
                    vector<uint8_t>& out) {
    const int k = static_cast<int>(qa.size());
    out.assign(k, 0);
    if (k == 0) return 0.f;
    if (static_cast<size_t>(k) > cap_q_) fail("too many queries for buffer");
    CUDA_CHECK(cudaMemcpy(d_qa_, qa.data(), k * sizeof(vid_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qb_, qb.data(), k * sizeof(vid_t), cudaMemcpyHostToDevice));
    GpuTimer t; t.start();
    k_dsu_query<<<grid_for(k, 256), 256>>>(d_word_, d_qa_, d_qb_, k, d_qout_);
    CUDA_CHECK_LAST();
    float ms = t.stop_ms();
    CUDA_CHECK(cudaMemcpy(out.data(), d_qout_, k * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    return ms;
}

vid_t GpuDSU::components() const {
    int c = 0;
    CUDA_CHECK(cudaMemcpy(&c, d_components_, sizeof(int), cudaMemcpyDeviceToHost));
    return c;
}

void GpuDSU::roots(vector<vid_t>& out) const {
    vid_t* d_out = dmalloc<vid_t>(n_);
    k_dsu_roots<<<grid_for(n_, 256), 256>>>(d_word_, n_, d_out);
    CUDA_CHECK_LAST();
    out.resize(n_);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, n_ * sizeof(vid_t), cudaMemcpyDeviceToHost));
    dfree(d_out, n_);
}
