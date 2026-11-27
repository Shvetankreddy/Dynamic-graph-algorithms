#include "common.cuh"

#include <algorithm>
#include <fstream>
#include <functional>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <unordered_map>
#include <utility>

using namepsace std;
// CSR construction  
CSR build_csr(vid_t n, const vector<Edge>& edges, bool directed) {
    CSR g;
    g.n = n;
    const size_t arcs = directed ? edges.size() : 2 * edges.size();
    if (arcs > static_cast<size_t>(INT32_MAX)) fail("graph too large for 32-bit edge offsets");
    g.m = static_cast<eid_t>(arcs);
    g.row_ptr.assign(static_cast<size_t>(n) + 1, 0);
    g.col.resize(arcs);
    g.w.resize(arcs);

    for (const Edge& e : edges) {
        g.row_ptr[e.u + 1]++;
        if (!directed) g.row_ptr[e.v + 1]++;
    }
    for (vid_t i = 0; i < n; ++i) g.row_ptr[i + 1] += g.row_ptr[i];

    vector<eid_t> pos(g.row_ptr.begin(), g.row_ptr.end() - 1);
    for (const Edge& e : edges) {
        eid_t p = pos[e.u]++;
        g.col[p] = e.v;
        g.w[p]   = e.w;
        if (!directed) {
            p = pos[e.v]++;
            g.col[p] = e.u;
            g.w[p]   = e.w;
        }
    }
    return g;
}

vid_t max_degree_vertex(const CSR& g) {
    vid_t best = 0;
    eid_t best_deg = -1;
    for (vid_t u = 0; u < g.n; ++u) {
        eid_t d = g.row_ptr[u + 1] - g.row_ptr[u];
        if (d > best_deg) { best_deg = d; best = u; }
    }
    return best;
}


// HostDynGraph  (from graph_gen.cpp)


HostDynGraph::HostDynGraph(const EdgeList& initial)
    : directed_(initial.directed),
      base_(build_csr(initial.n, initial.edges, initial.directed)),
      delta_(static_cast<size_t>(initial.n)) {}

void HostDynGraph::insert_edges(const vector<Edge>& batch) {
    for (const Edge& e : batch) {
        delta_[e.u].push_back({e.v, e.w});
        ++m_delta_;
        if (!directed_) {
            delta_[e.v].push_back({e.u, e.w});
            ++m_delta_;
        }
    }
}

bool HostDynGraph::has_arc(vid_t u, vid_t v, weight_t w) const {
    bool found = false;
    for_each_out(u, [&](vid_t x, weight_t wx) { if (x == v && wx == w) found = true; });
    return found;
}


// Graph generators  (from graph_gen.cpp)


EdgeList gen_uniform(vid_t n, size_t m, weight_t max_w, uint64_t seed, bool directed) {
    if (n < 2) fail("uniform graph needs n >= 2");
    EdgeList el;
    el.n = n; el.directed = directed;
    el.edges.reserve(m);
    mt19937_64 rng(seed);
    uniform_int_distribution<vid_t>    vd(0, n - 1);
    uniform_int_distribution<weight_t> wd(1, max_w);
    while (el.edges.size() < m) {
        vid_t u = vd(rng), v = vd(rng);
        if (u == v) continue;
        el.edges.push_back({u, v, wd(rng)});
    }
    return el;
}

EdgeList gen_rmat(int scale, int edge_factor, weight_t max_w, uint64_t seed, bool directed,
                  double a, double b, double c) {
    if (scale < 1 || scale > 30) fail("rmat scale must be in [1, 30]");
    EdgeList el;
    el.n = static_cast<vid_t>(1) << scale;
    el.directed = directed;
    const size_t m = static_cast<size_t>(edge_factor) * static_cast<size_t>(el.n);
    el.edges.reserve(m);

    mt19937_64 rng(seed);
    uniform_real_distribution<double>  ud(0.0, 1.0);
    uniform_int_distribution<weight_t> wd(1, max_w);

    vector<vid_t> perm(el.n);
    iota(perm.begin(), perm.end(), 0);
    shuffle(perm.begin(), perm.end(), rng);

    const double ab = a + b, abc = a + b + c;
    while (el.edges.size() < m) {
        vid_t u = 0, v = 0;
        for (int level = 0; level < scale; ++level) {
            double r = ud(rng);
            int bu, bv;
            if      (r < a)   { bu = 0; bv = 0; }
            else if (r < ab)  { bu = 0; bv = 1; }
            else if (r < abc) { bu = 1; bv = 0; }
            else              { bu = 1; bv = 1; }
            u = (u << 1) | bu;
            v = (v << 1) | bv;
        }
        if (u == v) continue;
        el.edges.push_back({perm[u], perm[v], wd(rng)});
    }
    return el;
}

EdgeList gen_grid(int rows, int cols, weight_t max_w, uint64_t seed, bool directed) {
    if (rows < 1 || cols < 1) fail("grid needs rows, cols >= 1");
    const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    if (n > static_cast<size_t>(INT32_MAX)) fail("grid too large");
    EdgeList el;
    el.n = static_cast<vid_t>(n); el.directed = directed;
    mt19937_64 rng(seed);
    uniform_int_distribution<weight_t> wd(1, max_w);
    auto id = [cols](int r, int c) { return static_cast<vid_t>(r * cols + c); };
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (c + 1 < cols) el.edges.push_back({id(r,c), id(r,c+1), wd(rng)});
            if (r + 1 < rows) el.edges.push_back({id(r,c), id(r+1,c), wd(rng)});
            if (directed) {
                if (c + 1 < cols) el.edges.push_back({id(r,c+1), id(r,c), wd(rng)});
                if (r + 1 < rows) el.edges.push_back({id(r+1,c), id(r,c), wd(rng)});
            }
        }
    }
    return el;
}

void split_initial_and_stream(const EdgeList& full, double init_frac, uint64_t seed,
                               EdgeList& initial, vector<Edge>& stream) {
    if (init_frac < 0.0 || init_frac > 1.0) fail("init-frac must be in [0, 1]");
    vector<Edge> shuffled = full.edges;
    mt19937_64 rng(seed ^ 0x9E3779B97F4A7C15ull);
    shuffle(shuffled.begin(), shuffled.end(), rng);
    const size_t k = static_cast<size_t>(init_frac * static_cast<double>(shuffled.size()));

    initial.n = full.n; initial.directed = full.directed;
    initial.edges.assign(shuffled.begin(), shuffled.begin() + k);
    stream.assign(shuffled.begin() + k, shuffled.end());
}


// File loaders  (from graph_io.cpp)


// SNAP edge list: "u v" or "u v w" per line; '#' or '%' lines are comments.
// Arbitrary vertex IDs are remapped to 0..n-1.
// If a weight column is missing or <= 0, weights are drawn uniformly from [1, max_w]
// using a seeded RNG, making the assignment deterministic and reproducible.
EdgeList load_snap(const string& path, weight_t max_w, uint64_t seed, bool directed) {
    ifstream in(path);
    if (!in) fail("cannot open " + path);

    EdgeList el;
    el.directed = directed;
    unordered_map<long long, vid_t> remap;
    auto get_id = [&](long long raw) {
        auto it = remap.find(raw);
        if (it != remap.end()) return it->second;
        vid_t id = static_cast<vid_t>(remap.size());
        remap.emplace(raw, id);
        return id;
    };

    mt19937_64 rng(seed);
    uniform_int_distribution<weight_t> wd(1, max_w);
    string line;
    while (getline(in, line)) {
        if (line.empty() || line[0] == '#' || line[0] == '%') continue;
        istringstream ss(line);
        long long a, b;
        if (!(ss >> a >> b)) continue;
        long long wraw;
        weight_t w = (ss >> wraw && wraw > 0) ? static_cast<weight_t>(wraw) : wd(rng);
        if (a == b) continue;   // drop self-loops
        vid_t u = get_id(a), v = get_id(b);
        el.edges.push_back({u, v, w});
    }
    el.n = static_cast<vid_t>(remap.size());
    if (el.n == 0) fail("no edges read from " + path);
    return el;
}

// DIMACS .gr shortest-path format: "p sp n m" header + "a u v w" arc lines (1-based IDs).
EdgeList load_dimacs(const string& path, bool directed) {
    ifstream in(path);
    if (!in) fail("cannot open " + path);

    EdgeList el;
    el.directed = directed;
    string line;
    while (getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == 'p') {
            istringstream ss(line);
            string p, sp; long long n, m;
            ss >> p >> sp >> n >> m;
            el.n = static_cast<vid_t>(n);
            el.edges.reserve(static_cast<size_t>(m));
        } else if (line[0] == 'a') {
            istringstream ss(line);
            char a; long long u, v, w;
            ss >> a >> u >> v >> w;
            if (u == v) continue;
            el.edges.push_back({static_cast<vid_t>(u-1), static_cast<vid_t>(v-1),
                                 static_cast<weight_t>(w > 0 ? w : 1)});
        }
    }
    if (el.n == 0) fail("missing 'p sp n m' header in " + path);
    return el;
}


// CpuDSU  (from cpu_ref.cpp)


void CpuDSU::reset(vid_t n) {
    parent_.resize(n);
    rank_.assign(n, 0);
    for (vid_t i = 0; i < n; ++i) parent_[i] = i;
    components_ = n;
}

vid_t CpuDSU::find(vid_t x) {
    vid_t root = x;
    while (parent_[root] != root) root = parent_[root];
    while (parent_[x] != root) {   // full path compression
        vid_t next = parent_[x];
        parent_[x] = root;
        x = next;
    }
    return root;
}

bool CpuDSU::unite(vid_t a, vid_t b) {
    a = find(a); b = find(b);
    if (a == b) return false;
    if (rank_[a] < rank_[b]) swap(a, b);
    parent_[b] = a;
    if (rank_[a] == rank_[b]) ++rank_[a];
    --components_;
    return true;
}

vid_t cpu_cc_recompute(const HostDynGraph& g) {
    CpuDSU dsu(g.n());
    for (vid_t u = 0; u < g.n(); ++u)
        g.for_each_out(u, [&](vid_t v, weight_t) {
            if (g.directed() || u < v) dsu.unite(u, v);
        });
    return dsu.components();
}


// CPU Dijkstra  (from cpu_ref.cpp)


using HeapItem = pair<dist_t, vid_t>;
using MinHeap  = priority_queue<HeapItem, vector<HeapItem>, greater<HeapItem>>;

static void dijkstra_propagate(const HostDynGraph& g, MinHeap& heap,
                                vector<dist_t>& dist, size_t* settled_changes) {
    while (!heap.empty()) {
        auto [d, u] = heap.top(); heap.pop();
        if (d != dist[u]) continue;   // stale entry
        if (settled_changes) ++(*settled_changes);
        g.for_each_out(u, [&](vid_t v, weight_t w) {
            dist_t nd = d + w;
            if (nd < dist[v]) { dist[v] = nd; heap.push({nd, v}); }
        });
    }
}

void cpu_dijkstra(const HostDynGraph& g, vid_t src, vector<dist_t>& dist) {
    dist.assign(g.n(), DIST_INF);
    dist[src] = 0;
    MinHeap heap;
    heap.push({0, src});
    dijkstra_propagate(g, heap, dist, nullptr);
}

size_t cpu_sssp_incremental(const HostDynGraph& g, const vector<Edge>& batch,
                             vector<dist_t>& dist) {
    MinHeap heap;
    auto seed = [&](vid_t a, vid_t b, weight_t w) {
        if (dist[a] == DIST_INF) return;
        dist_t nd = dist[a] + w;
        if (nd < dist[b]) { dist[b] = nd; heap.push({nd, b}); }
    };
    for (const Edge& e : batch) {
        seed(e.u, e.v, e.w);
        if (!g.directed()) seed(e.v, e.u, e.w);
    }
    size_t changed = 0;
    dijkstra_propagate(g, heap, dist, &changed);
    return changed;
}
