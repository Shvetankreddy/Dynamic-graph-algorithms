#include <algorithm>
#include <cstdarg>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "common.cuh"
using namespace std;

// Command-line options

struct Options {
    string graph       = "uniform";  // uniform | rmat | grid | snap | dimacs
    string file;
    long long   n           = 100000;
    long long   m           = 1000000;
    int         scale       = 16;
    int         edge_factor = 16;
    int         rows        = 512;
    int         cols        = 512;
    bool        directed    = false;
    weight_t    max_w       = 100;
    double      init_frac   = 0.5;
    long long   batch_size  = 1000;
    long long   num_batches = 10;     // -1 = whole stream
    long long   source      = -1;     // -1 = max-degree vertex of initial graph
    int         verify_every   = 1;   // 0 = never verify
    int         baseline_every = 1;   // 0 = never run baselines
    bool        run_bf      = true;
    int         bf_max_iters = -1;
    bool        load_balance = true;
    int         block       = 256;
    int         queries     = 1000;
    double      merge_frac  = 0.1;
    string mode        = "all";  // all | cc | sssp
    string out;                 
    bool        out_explicit = false;
    bool        no_out      = false;
    string csv;                 
    string tag         = "run";
    uint64_t    seed        = 42;
    int         device      = 0;
    bool        quiet       = false;
};

static void usage() {
    printf(
        "Usage: dynamic_graph [dsu|sssp] FILE [options]\n"
        "   or: dynamic_graph [options]\n"
        "\n"
        "Positional commands (set mode and file at once):\n"
        "  dsu  FILE    Run only DSU (connected components) on FILE\n"
        "  sssp FILE    Run only SSSP on FILE\n"
        "\n"
        "Graph source\n"
        "  --graph uniform|rmat|grid|snap|dimacs   (default uniform)\n"
        "  --n N --m M                 uniform graph size\n"
        "  --scale S --edge-factor E   rmat: 2^S vertices, E*2^S edges\n"
        "  --rows R --cols C           grid size\n"
        "  --file PATH                 input file for snap / dimacs\n"
        "  --directed                  treat graph as directed (default undirected)\n"
        "  --max-w W                   max edge weight (default 100)\n"
        "Update stream\n"
        "  --init-frac F               fraction of edges in the initial graph (default 0.5)\n"
        "  --batch-size B              insertions per batch (default 1000)\n"
        "  --num-batches K             batches to run, -1 for whole stream (default 10)\n"
        "Algorithms\n"
        "  --mode all|cc|sssp          which analyses to run (default all)\n"
        "  --source S                  SSSP source, -1 = max-degree vertex (default -1)\n"
        "  --no-lb                     disable degree-aware load balancing\n"
        "  --block T                   threads per block (default 256)\n"
        "  --merge-frac F              merge delta into base CSR when delta > F*base (0.1)\n"
        "  --queries Q                 connectivity queries per batch (default 1000)\n"
        "Measurement & Output\n"
        "  --verify-every K            verify every K batches, 0 = off (default 1)\n"
        "  --baseline-every K          run static baselines every K batches, 0 = off (1)\n"
        "  --no-bf                     skip the Bellman-Ford baseline\n"
        "  --bf-max-iters I            cap Bellman-Ford iterations\n"
        "  --out PATH                  write formatted .txt table report to PATH\n"
        "  --no-out                    do not write any .txt file (stdout only)\n"
        "  --csv PATH                  also write a compact mode CSV to PATH\n"
        "  --tag NAME                  label stored in CSV\n"
        "  --seed S --device D --quiet\n");
}

static Options parse_args(int argc, char** argv) {
    Options o;

    int arg_start = 1;
    if (argc >= 3) {
        string cmd = argv[1];
        if (cmd == "dsu") {
            o.mode  = "cc";
            o.graph = "snap";
            o.file  = argv[2];
            arg_start = 3;
        } else if (cmd == "sssp") {
            o.mode  = "sssp";
            o.graph = "snap";
            o.file  = argv[2];
            arg_start = 3;
        }
    }

    auto need = [&](int& i) -> string {
        if (i + 1 >= argc) fail(string("missing value for ") + argv[i]);
        return argv[++i];
    };
    for (int i = arg_start; i < argc; ++i) {
        string a = argv[i];
        if      (a == "--help"    || a == "-h")  { usage(); exit(0); }
        else if (a == "--graph")       o.graph        = need(i);
        else if (a == "--file")        o.file         = need(i);
        else if (a == "--n")           o.n            = stoll(need(i));
        else if (a == "--m")           o.m            = stoll(need(i));
        else if (a == "--scale")       o.scale        = stoi(need(i));
        else if (a == "--edge-factor") o.edge_factor  = stoi(need(i));
        else if (a == "--rows")        o.rows         = stoi(need(i));
        else if (a == "--cols")        o.cols         = stoi(need(i));
        else if (a == "--directed")    o.directed     = true;
        else if (a == "--max-w")       o.max_w        = static_cast<weight_t>(stoul(need(i)));
        else if (a == "--init-frac")   o.init_frac    = stod(need(i));
        else if (a == "--batch-size")  o.batch_size   = stoll(need(i));
        else if (a == "--num-batches") o.num_batches  = stoll(need(i));
        else if (a == "--source")      o.source       = stoll(need(i));
        else if (a == "--verify-every")   o.verify_every   = stoi(need(i));
        else if (a == "--baseline-every") o.baseline_every = stoi(need(i));
        else if (a == "--no-bf")       o.run_bf       = false;
        else if (a == "--bf-max-iters")o.bf_max_iters = stoi(need(i));
        else if (a == "--no-lb")       o.load_balance = false;
        else if (a == "--block")       o.block        = stoi(need(i));
        else if (a == "--queries")     o.queries      = stoi(need(i));
        else if (a == "--merge-frac")  o.merge_frac   = stod(need(i));
        else if (a == "--mode") {
            o.mode = need(i);
            if (o.mode == "dsu") o.mode = "cc";
        }
        else if (a == "--out")         { o.out = need(i); o.out_explicit = true; }
        else if (a == "--no-out")      o.no_out       = true;
        else if (a == "--csv")         o.csv          = need(i);
        else if (a == "--tag")         o.tag          = need(i);
        else if (a == "--seed")        o.seed         = stoull(need(i));
        else if (a == "--device")      o.device       = stoi(need(i));
        else if (a == "--quiet")       o.quiet        = true;
        else { usage(); fail("unknown option " + a); }
    }
    if (o.batch_size < 1) fail("batch-size must be >= 1");
    if (o.block < 32 || o.block > 1024 || o.block % 32 != 0)
        fail("block must be a multiple of 32 in [32, 1024]");
    if (o.max_w < 1) fail("max-w must be >= 1");
    if (o.mode != "all" && o.mode != "cc" && o.mode != "sssp")
        fail("mode must be all, cc or sssp");
    return o;
}

static EdgeList make_graph(const Options& o) {
    if (o.graph == "uniform") return gen_uniform(static_cast<vid_t>(o.n), o.m, o.max_w, o.seed, o.directed);
    if (o.graph == "rmat")    return gen_rmat(o.scale, o.edge_factor, o.max_w, o.seed, o.directed);
    if (o.graph == "grid")    return gen_grid(o.rows, o.cols, o.max_w, o.seed, o.directed);
    if (o.graph == "snap")    return load_snap(o.file, o.max_w, o.seed, o.directed);
    if (o.graph == "dimacs")  return load_dimacs(o.file, o.directed);
    fail("unknown graph type " + o.graph);
    return {};
}


// Verification helpers

static bool verify_partition(const vector<vid_t>& gpu_root, CpuDSU& cpu, vid_t gpu_comp) {
    if (gpu_comp != cpu.components()) {
        printf("  [verify] component count mismatch: gpu=%d cpu=%d\n",
                    gpu_comp, cpu.components());
        return false;
    }
    const vid_t n = static_cast<vid_t>(gpu_root.size());
    vector<vid_t> map(n, -1);
    for (vid_t v = 0; v < n; ++v) {
        vid_t c = cpu.find(v);
        if (map[c] == -1) map[c] = gpu_root[v];
        else if (map[c] != gpu_root[v]) {
            printf("  [verify] vertex %d: gpu set differs from cpu set\n", v);
            return false;
        }
    }
    return true;
}

static bool verify_dist(const char* what, const vector<dist_t>& got,
                        const vector<dist_t>& ref) {
    for (size_t v = 0; v < ref.size(); ++v) {
        if (got[v] != ref[v]) {
            printf("  [verify] %s: dist mismatch at %zu: got %u expected %u\n",
                        what, v, got[v], ref[v]);
            return false;
        }
    }
    return true;
}

static bool verify_tree(const HostDynGraph& g, vid_t src,
                        const vector<dist_t>& dist, const vector<vid_t>& pred) {
    const vid_t n = g.n();
    vector<uint8_t> ok(n, 0);
    for (vid_t u = 0; u < n; ++u) {
        if (dist[u] == DIST_INF) continue;
        g.for_each_out(u, [&](vid_t v, weight_t w) {
            if (pred[v] == u && dist[v] != DIST_INF && dist[u] + w == dist[v]) ok[v] = 1;
        });
    }
    for (vid_t v = 0; v < n; ++v) {
        if (v == src || dist[v] == DIST_INF) continue;
        if (!ok[v]) {
            printf("  [verify] invalid tree edge at vertex %d (pred %d)\n", v, pred[v]);
            return false;
        }
    }
    return true;
}

struct DsuRow {
    long long batch = 0;
    int new_edges = 0;
    size_t total_edges = 0;
    vid_t comps_before = 0;
    vid_t comps_after = 0;
    double gpu_dsu_ms = 0.0;
    double cpu_dsu_ms = 0.0;
    string check;
};

struct SsspRow {
    long long batch = 0;
    int new_edges = 0;
    size_t total_edges = 0;
    size_t vertices_updated = 0;
    double gpu_sssp_ms = 0.0;
    double cpu_incr_ms = 0.0;
    double cpu_dijk_ms = -1.0;
    string check;
};

static string fmt_num(double x, int prec = 3) {
    if (x < 0.0) return "-";
    char buf[64];
    snprintf(buf, sizeof(buf), "%.*f", prec, x);
    return buf;
}

static string fmt_speedup(double cpu_ms, double gpu_ms) {
    if (cpu_ms < 0.0 || gpu_ms <= 0.0) return "-";
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2fx", cpu_ms / gpu_ms);
    return buf;
}

static string format_str(const char* fmt_spec, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt_spec);
    vsnprintf(buf, sizeof(buf), fmt_spec, args);
    va_end(args);
    return string(buf);
}

static void emit_line(ofstream* file, bool quiet, const string& line) {
    if (!quiet) printf("%s\n", line.c_str());
    if (file && file->is_open()) *file << line << "\n";
}

static string open_txt_report(const string& path, bool explicit_path, ofstream& ofs) {
    if (path.empty()) return "";
    ofs.open(path, ios::trunc);
    if (!ofs) {
        if (explicit_path) fail("cannot open output report " + path);
        return "";
    }
    return path;
}

static void write_dsu_table(const vector<DsuRow>& rows, const string& header_info,
                            vid_t init_comps, float init_gpu_ms, const string& init_check,
                            double mean_e2e_ms, double peak_mb, bool all_ok, int verify_every,
                            const string& out_path, bool out_explicit, bool quiet) {
    ofstream ofs;
    string saved_path = open_txt_report(out_path, out_explicit, ofs);
    ofstream* fp = ofs.is_open() ? &ofs : nullptr;

    emit_line(fp, quiet, "=== DSU MODE (Weakly Connected Components) ===");
    emit_line(fp, quiet, header_info);
    emit_line(fp, quiet, format_str("Initial connectivity: %d components (GPU build: %.3f ms)%s",
                                    init_comps, init_gpu_ms, init_check.c_str()));
    emit_line(fp, quiet, "");
    emit_line(fp, quiet, format_str("%5s | %9s | %11s | %28s | %10s | %10s | %8s | %5s",
                                    "batch", "new_edges", "total_edges",
                                    "components (b->a / merged)",
                                    "gpu_dsu_ms", "cpu_dsu_ms", "speedup", "check"));
    emit_line(fp, quiet, "------+-----------+-------------+------------------------------+------------+------------+----------+------");

    double sum_gpu = 0.0, sum_cpu = 0.0;
    long long total_merged = 0;
    for (const auto& r : rows) {
        vid_t merged_c = r.comps_before - r.comps_after;
        total_merged += merged_c;
        sum_gpu += r.gpu_dsu_ms;
        sum_cpu += r.cpu_dsu_ms;
        string comp_str = format_str("%d -> %d (%d)", r.comps_before, r.comps_after, merged_c);
        string sp = fmt_speedup(r.cpu_dsu_ms, r.gpu_dsu_ms);
        emit_line(fp, quiet, format_str("%5lld | %9d | %11zu | %28s | %10.3f | %10.4f | %8s | %5s",
                                        r.batch, r.new_edges, r.total_edges, comp_str.c_str(),
                                        r.gpu_dsu_ms, r.cpu_dsu_ms, sp.c_str(), r.check.c_str()));
    }

    const size_t nb = rows.size();
    vid_t final_comps = nb > 0 ? rows.back().comps_after : init_comps;
    emit_line(fp, quiet, "");
    emit_line(fp, quiet, format_str("DSU Summary (%zu batches):", nb));
    emit_line(fp, quiet, format_str("  components (initial -> final) : %d -> %d (%lld merged)",
                                    init_comps, final_comps, total_merged));
    if (nb > 0) {
        double mean_gpu = sum_gpu / nb;
        double mean_cpu = sum_cpu / nb;
        emit_line(fp, quiet, format_str("  mean GPU DSU kernel           : %.3f ms/batch (e2e incl. H2D+CSR: %.3f ms)",
                                        mean_gpu, mean_e2e_ms));
        emit_line(fp, quiet, format_str("  mean CPU DSU update           : %.4f ms/batch (speedup: %s)",
                                        mean_cpu, fmt_speedup(mean_cpu, mean_gpu).c_str()));
    }
    emit_line(fp, quiet, format_str("  tracked GPU memory (peak)     : %.1f MB", peak_mb));
    if (!saved_path.empty())
        emit_line(fp, quiet, format_str("  saved report                  : %s", saved_path.c_str()));
    emit_line(fp, quiet, format_str("  verification                  : %s",
                                    verify_every > 0 ? (all_ok ? "ALL PASS" : "FAILURES FOUND") : "disabled"));
}

static void write_sssp_table(const vector<SsspRow>& rows, const string& header_info,
                             vid_t src, size_t init_reach, float init_gpu_ms, int init_iters,
                             const string& init_check, double mean_e2e_ms, double peak_mb,
                             bool all_ok, int verify_every, const string& out_path,
                             bool out_explicit, bool quiet) {
    ofstream ofs;
    string saved_path = open_txt_report(out_path, out_explicit, ofs);
    ofstream* fp = ofs.is_open() ? &ofs : nullptr;

    emit_line(fp, quiet, "=== SSSP MODE (Single-Source Shortest Path) ===");
    emit_line(fp, quiet, header_info);
    emit_line(fp, quiet, format_str("Initial SSSP (source=%d): %zu reachable vertices (GPU frontier: %.3f ms, %d iters)%s",
                                    src, init_reach, init_gpu_ms, init_iters, init_check.c_str()));
    emit_line(fp, quiet, "");
    emit_line(fp, quiet, format_str("%5s | %9s | %11s | %16s | %11s | %11s | %11s | %8s | %5s",
                                    "batch", "new_edges", "total_edges", "vertices_updated",
                                    "gpu_sssp_ms", "cpu_incr_ms", "cpu_dijk_ms", "speedup", "check"));
    emit_line(fp, quiet, "------+-----------+-------------+------------------+-------------+-------------+-------------+----------+------");

    double sum_gpu = 0.0, sum_cpu_incr = 0.0, sum_dijk = 0.0;
    size_t dijk_cnt = 0;
    size_t total_updated = 0;
    for (const auto& r : rows) {
        total_updated += r.vertices_updated;
        sum_gpu      += r.gpu_sssp_ms;
        sum_cpu_incr += r.cpu_incr_ms;
        if (r.cpu_dijk_ms >= 0.0) { sum_dijk += r.cpu_dijk_ms; ++dijk_cnt; }
        double ref_cpu = (r.cpu_dijk_ms >= 0.0) ? r.cpu_dijk_ms : r.cpu_incr_ms;
        string sp = fmt_speedup(ref_cpu, r.gpu_sssp_ms);
        emit_line(fp, quiet, format_str("%5lld | %9d | %11zu | %16zu | %11.3f | %11.4f | %11s | %8s | %5s",
                                        r.batch, r.new_edges, r.total_edges, r.vertices_updated,
                                        r.gpu_sssp_ms, r.cpu_incr_ms, fmt_num(r.cpu_dijk_ms, 4).c_str(),
                                        sp.c_str(), r.check.c_str()));
    }

    const size_t nb = rows.size();
    emit_line(fp, quiet, "");
    emit_line(fp, quiet, format_str("SSSP Summary (%zu batches):", nb));
    emit_line(fp, quiet, format_str("  total vertices updated        : %zu", total_updated));
    if (nb > 0) {
        double mean_gpu = sum_gpu / nb;
        double mean_ci  = sum_cpu_incr / nb;
        emit_line(fp, quiet, format_str("  mean GPU incremental SSSP     : %.3f ms/batch (e2e incl. H2D+CSR: %.3f ms)",
                                        mean_gpu, mean_e2e_ms));
        emit_line(fp, quiet, format_str("  mean CPU incremental SSSP     : %.4f ms/batch", mean_ci));
        if (dijk_cnt > 0) {
            double mean_dijk = sum_dijk / dijk_cnt;
            emit_line(fp, quiet, format_str("  mean CPU Dijkstra recompute   : %.4f ms/batch (speedup vs Dijkstra: %s)",
                                            mean_dijk, fmt_speedup(mean_dijk, mean_gpu).c_str()));
        }
    }
    emit_line(fp, quiet, format_str("  tracked GPU memory (peak)     : %.1f MB", peak_mb));
    if (!saved_path.empty())
        emit_line(fp, quiet, format_str("  saved report                  : %s", saved_path.c_str()));
    emit_line(fp, quiet, format_str("  verification                  : %s",
                                    verify_every > 0 ? (all_ok ? "ALL PASS" : "FAILURES FOUND") : "disabled"));
}

int main(int argc, char** argv) {
    Options o = parse_args(argc, argv);
    const bool do_cc   = (o.mode == "all" || o.mode == "cc");
    const bool do_sssp = (o.mode == "all" || o.mode == "sssp");

    // ---- device ----
    CUDA_CHECK(cudaSetDevice(o.device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, o.device));
    CUDA_CHECK(cudaFree(0));
    size_t free0 = 0, total0 = 0;
    CUDA_CHECK(cudaMemGetInfo(&free0, &total0));
    string gpu_line = format_str("GPU %d: %s (sm_%d%d, %d SMs, %.1f GB)",
                                      o.device, prop.name, prop.major, prop.minor,
                                      prop.multiProcessorCount, total0 / 1073741824.0);

    // - graph and update stream -
    HostTimer ht; ht.start();
    EdgeList full = make_graph(o);
    EdgeList initial;
    vector<Edge> stream;
    split_initial_and_stream(full, o.init_frac, o.seed, initial, stream);
    full.edges.clear(); full.edges.shrink_to_fit();

    long long avail_batches =
        (static_cast<long long>(stream.size()) + o.batch_size - 1) / o.batch_size;
    long long nb = (o.num_batches < 0) ? avail_batches : min(o.num_batches, avail_batches);

    HostDynGraph hg(initial);
    const vid_t n = hg.n();
    vid_t src = (o.source < 0) ? max_degree_vertex(hg.base()) : static_cast<vid_t>(o.source);
    if (src >= n) fail("source out of range");

    string src_name = o.file.empty() ? o.graph : o.file;
    string graph_line = format_str("%s | Graph %s: n=%d, initial_edges=%zu, stream_edges=%zu, batches=%lld x %lld, %s (built in %.1f ms)",
                                        gpu_line.c_str(), src_name.c_str(), n,
                                        initial.edges.size(), stream.size(), nb, o.batch_size,
                                        o.directed ? "directed" : "undirected", ht.stop_ms());

    // GPU structures 
    DeviceDynGraph dg;
    dg.init(hg.base(), o.directed, static_cast<size_t>(o.batch_size), o.merge_frac);

    GpuDSU  dsu, dsu_static;
    GpuSSSP sssp, sssp_static;
    CpuDSU  cdsu;
    vector<dist_t> cpu_dist, ref_dist, gpu_dist, tmp_dist;
    vector<vid_t>  gpu_pred, gpu_roots;
    bool ok_cc_all = true, ok_sssp_all = true;

    vid_t init_comps = 0;
    float init_cc_ms = 0.f;
    string init_cc_check;
    if (do_cc) {
        dsu.init(n, o.queries);
        init_cc_ms = dsu.recompute_from_graph(dg.view(), o.directed);
        cdsu.reset(n);
        for (const Edge& e : initial.edges) cdsu.unite(e.u, e.v);
        init_comps = dsu.components();
        if (o.verify_every > 0) {
            dsu.roots(gpu_roots);
            bool ok = verify_partition(gpu_roots, cdsu, init_comps);
            init_cc_check = ok ? " | Verify: PASS" : " | Verify: FAIL";
            ok_cc_all &= ok;
        }
        if (o.baseline_every > 0) dsu_static.init(n, 0);
    }

    size_t init_reach = 0;
    float init_sssp_ms = 0.f;
    int init_sssp_iters = 0;
    string init_sssp_check;
    if (do_sssp) {
        sssp.init(n, o.block, o.load_balance);
        SSSPStats s0 = sssp.run_frontier(dg.view(), src);
        init_sssp_ms = s0.total_ms;
        init_sssp_iters = s0.iterations;
        cpu_dijkstra(hg, src, cpu_dist);
        init_reach = count_if(cpu_dist.begin(), cpu_dist.end(),
                                   [](dist_t d) { return d != DIST_INF; });
        if (o.verify_every > 0) {
            sssp.distances(gpu_dist, &gpu_pred);
            bool ok = verify_dist("initial sssp", gpu_dist, cpu_dist) &&
                      verify_tree(hg, src, gpu_dist, gpu_pred);
            init_sssp_check = ok ? " | Verify: PASS" : " | Verify: FAIL";
            ok_sssp_all &= ok;
        }
        if (o.baseline_every > 0) sssp_static.init(n, o.block, o.load_balance);
    }

    // ---- Optional CSV ----
    ofstream csv;
    if (!o.csv.empty()) {
        csv.open(o.csv, ios::trunc);
        if (!csv) fail("cannot open csv " + o.csv);
        if (o.mode == "cc") {
            csv << "tag,batch,new_edges,total_edges,comps_before,comps_after,merged_comps,gpu_dsu_ms,cpu_dsu_ms,check\n";
        } else if (o.mode == "sssp") {
            csv << "tag,batch,new_edges,total_edges,vertices_updated,gpu_sssp_ms,cpu_incr_ms,cpu_dijk_ms,check\n";
        } else {
            csv << "tag,batch,new_edges,total_edges,comps_before,comps_after,gpu_dsu_ms,cpu_dsu_ms,vertices_updated,gpu_sssp_ms,cpu_incr_ms,cpu_dijk_ms,verify_cc,verify_sssp\n";
        }
    }

    mt19937_64 qrng(o.seed + 7);
    uniform_int_distribution<vid_t> qd(0, n - 1);
    vector<vid_t>   qa(o.queries), qb(o.queries);
    vector<uint8_t> qout;

    vector<DsuRow>  dsu_rows;
    vector<SsspRow> sssp_rows;
    double sum_e2e_dsu = 0.0, sum_e2e_sssp = 0.0;
    vid_t prev_comps = init_comps;

    // ---- batch loop ----
    for (long long b = 0; b < nb; ++b) {
        const size_t lo = static_cast<size_t>(b * o.batch_size);
        const size_t hi = min(stream.size(), lo + static_cast<size_t>(o.batch_size));
        vector<Edge> batch(stream.begin() + lo, stream.begin() + hi);
        const int k = static_cast<int>(batch.size());

        // GPU graph update + algorithms
        HostTimer e2e_graph; e2e_graph.start();
        float h2d = dg.upload_batch(batch);
        bool merged = false;
        float upd = dg.apply_batch(&merged);
        (void)h2d; (void)upd;
        CUDA_CHECK(cudaDeviceSynchronize());
        double graph_ms = e2e_graph.stop_ms();

        float dsu_ms = -1.f;
        SSSPStats st;
        if (do_cc) {
            HostTimer t; t.start();
            dsu_ms = dsu.union_edges(dg.staged_u(), dg.staged_v(), k);
            CUDA_CHECK(cudaDeviceSynchronize());
            sum_e2e_dsu += graph_ms + t.stop_ms();
        }
        if (do_sssp) {
            HostTimer t; t.start();
            st = sssp.update_insertions(dg.view(), dg.staged_u(), dg.staged_v(),
                                        dg.staged_w(), k, o.directed);
            CUDA_CHECK(cudaDeviceSynchronize());
            sum_e2e_sssp += graph_ms + t.stop_ms();
        }

        // CPU incremental references
        hg.insert_edges(batch);
        double cpu_dsu_ms = -1.0, cpu_sssp_ms = -1.0;
        size_t cpu_affected = 0;
        if (do_cc) {
            HostTimer t; t.start();
            for (const Edge& e : batch) cdsu.unite(e.u, e.v);
            cpu_dsu_ms = t.stop_ms();
        }
        if (do_sssp) {
            HostTimer t; t.start();
            cpu_affected = cpu_sssp_incremental(hg, batch, cpu_dist);
            cpu_sssp_ms  = t.stop_ms();
        }

        // Connectivity queries
        vid_t comps = -1;
        if (do_cc) {
            comps = dsu.components();
            for (int i = 0; i < o.queries; ++i) { qa[i] = qd(qrng); qb[i] = qd(qrng); }
            dsu.query(qa, qb, qout);
        }

        // Static baselines & Verification
        const bool run_base   = o.baseline_every > 0 && (b % o.baseline_every == 0);
        const bool run_verify = o.verify_every   > 0 && (b % o.verify_every   == 0);
        double dijk_ms = -1.0;
        bool have_ref = false, ok_cc = true, ok_sssp = true;
        bool base_cc_ok = true, base_sssp_ok = true;

        if (run_base && do_cc) {
            dsu_static.recompute_from_graph(dg.view(), o.directed);
            vid_t c = cpu_cc_recompute(hg);
            if (dsu_static.components() != c) {
                printf("  [verify] static GPU CC count %d != CPU recompute %d\n",
                            dsu_static.components(), c);
                base_cc_ok = false;
            }
        }
        if (do_sssp && (run_base || run_verify)) {
            HostTimer t; t.start();
            cpu_dijkstra(hg, src, ref_dist);
            dijk_ms = t.stop_ms();
            have_ref = true;
        }
        if (run_base && do_sssp) {
            sssp_static.run_frontier(dg.view(), src);
            sssp_static.distances(tmp_dist);
            base_sssp_ok &= verify_dist("static frontier", tmp_dist, ref_dist);
            if (o.run_bf) {
                sssp_static.run_bellman_ford(dg.view(), src, o.bf_max_iters);
                if (o.bf_max_iters < 0) {
                    sssp_static.distances(tmp_dist);
                    base_sssp_ok &= verify_dist("static bellman ford", tmp_dist, ref_dist);
                }
            }
        }

        string chk_cc = "-", chk_sssp = "-";
        if (run_verify) {
            if (do_cc) {
                dsu.roots(gpu_roots);
                ok_cc = verify_partition(gpu_roots, cdsu, comps);
                for (int i = 0; i < o.queries && ok_cc; ++i) {
                    bool expect = cdsu.find(qa[i]) == cdsu.find(qb[i]);
                    if (expect != (qout[i] != 0)) {
                        printf("  [verify] query (%d,%d) wrong\n", qa[i], qb[i]);
                        ok_cc = false;
                    }
                }
                ok_cc &= base_cc_ok;
                chk_cc = ok_cc ? "PASS" : "FAIL";
                ok_cc_all &= ok_cc;
            }
            if (do_sssp && have_ref) {
                sssp.distances(gpu_dist, &gpu_pred);
                ok_sssp = verify_dist("incremental gpu",  gpu_dist, ref_dist) &&
                          verify_dist("incremental cpu",  cpu_dist, ref_dist) &&
                          verify_tree(hg, src, gpu_dist, gpu_pred) && base_sssp_ok;
                chk_sssp = ok_sssp ? "PASS" : "FAIL";
                ok_sssp_all &= ok_sssp;
            }
        }

        if (do_cc) {
            dsu_rows.push_back({b, k, hg.num_edges(), prev_comps, comps, dsu_ms, cpu_dsu_ms, chk_cc});
            prev_comps = comps;
        }
        if (do_sssp) {
            sssp_rows.push_back({b, k, hg.num_edges(), cpu_affected, st.total_ms, cpu_sssp_ms, dijk_ms, chk_sssp});
        }

        if (csv.is_open()) {
            if (o.mode == "cc") {
                const auto& r = dsu_rows.back();
                csv << o.tag << ',' << b << ',' << k << ',' << r.total_edges << ','
                    << r.comps_before << ',' << r.comps_after << ',' << (r.comps_before - r.comps_after) << ','
                    << fmt_num(r.gpu_dsu_ms, 4) << ',' << fmt_num(r.cpu_dsu_ms, 4) << ',' << r.check << "\n";
            } else if (o.mode == "sssp") {
                const auto& r = sssp_rows.back();
                csv << o.tag << ',' << b << ',' << k << ',' << r.total_edges << ','
                    << r.vertices_updated << ',' << fmt_num(r.gpu_sssp_ms, 4) << ','
                    << fmt_num(r.cpu_incr_ms, 4) << ',' << fmt_num(r.cpu_dijk_ms, 4) << ',' << r.check << "\n";
            } else {
                const auto& rc = dsu_rows.back();
                const auto& rs = sssp_rows.back();
                csv << o.tag << ',' << b << ',' << k << ',' << rc.total_edges << ','
                    << rc.comps_before << ',' << rc.comps_after << ','
                    << fmt_num(rc.gpu_dsu_ms, 4) << ',' << fmt_num(rc.cpu_dsu_ms, 4) << ','
                    << rs.vertices_updated << ',' << fmt_num(rs.gpu_sssp_ms, 4) << ','
                    << fmt_num(rs.cpu_incr_ms, 4) << ',' << fmt_num(rs.cpu_dijk_ms, 4) << ','
                    << rc.check << ',' << rs.check << "\n";
            }
        }
    }
    if (csv.is_open()) csv.close();

    //  Output 2 separate clean tables (+ separate .txt files) 
    const double peak_mb = device_mem_stats().peak / 1048576.0;
    const double mean_e2e_dsu  = nb > 0 ? sum_e2e_dsu  / nb : 0.0;
    const double mean_e2e_sssp = nb > 0 ? sum_e2e_sssp / nb : 0.0;

    string dsu_out_path, sssp_out_path;
    if (!o.no_out) {
        if (o.out_explicit) {
            if (o.mode == "cc")        dsu_out_path  = o.out;
            else if (o.mode == "sssp") sssp_out_path = o.out;
            else {
                dsu_out_path  = o.out + ".dsu.txt";
                sssp_out_path = o.out + ".sssp.txt";
            }
        } else {
            if (do_cc)   dsu_out_path  = "results/dsu_results.txt";
            if (do_sssp) sssp_out_path = "results/sssp_results.txt";
        }
    }

    if (do_cc) {
        write_dsu_table(dsu_rows, graph_line, init_comps, init_cc_ms, init_cc_check,
                        mean_e2e_dsu, peak_mb, ok_cc_all, o.verify_every,
                        dsu_out_path, o.out_explicit, o.quiet);
    }
    if (do_cc && do_sssp && !o.quiet) printf("\n");
    if (do_sssp) {
        write_sssp_table(sssp_rows, graph_line, src, init_reach, init_sssp_ms, init_sssp_iters,
                         init_sssp_check, mean_e2e_sssp, peak_mb, ok_sssp_all, o.verify_every,
                         sssp_out_path, o.out_explicit, o.quiet);
    }

    return (ok_cc_all && ok_sssp_all) ? 0 : 1;
}
