#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -f prompt.txt [-c 4096] [-o expert-usage.csv]\n", argv[0]);
    LOG("\n");
}

// per-layer expert selection counts, keyed by layer index
struct expert_profiler {
    std::mutex mutex;

    std::map<int, std::vector<int64_t>> counts; // layer -> [n_expert] selection counts
    int64_t n_decisions = 0;                    // total (token, slot) routing decisions seen

    std::vector<char> ids_host; // scratch buffer for copying ids off the backend

    // blk.<il>.ffn_down_exps.weight, possibly with a "BACKEND#...#N" wrapper
    static int parse_layer(const char * name) {
        const char * p = strstr(name, "blk.");
        if (p == nullptr) {
            return -1;
        }
        return atoi(p + 4);
    }

    bool eval(ggml_tensor * t, bool ask) {
        if (t->op != GGML_OP_MUL_MAT_ID) {
            return false;
        }

        const ggml_tensor * src0 = t->src[0];

        // every routed FFN runs exactly one down-projection per layer, so counting only
        // the down_exps mul_mat_id gives one tally per routing decision (gate/up reuse the same ids)
        const bool is_down = strstr(src0->name, "down_exps") != nullptr;
        if (ask) {
            return is_down;
        }
        if (!is_down) {
            return false;
        }

        const int il = parse_layer(src0->name);
        if (il < 0) {
            return false;
        }

        const ggml_tensor * ids = t->src[2]; // [n_expert_used, n_tokens], i32
        const int64_t n_expert  = src0->ne[2];

        std::lock_guard<std::mutex> lock(mutex);

        ids_host.resize(ggml_nbytes(ids));
        ggml_backend_tensor_get(ids, ids_host.data(), 0, ggml_nbytes(ids));

        auto & layer_counts = counts[il];
        if (layer_counts.empty()) {
            layer_counts.resize(n_expert, 0);
        }

        for (int64_t tok = 0; tok < ids->ne[1]; ++tok) {
            for (int64_t slot = 0; slot < ids->ne[0]; ++slot) {
                const int32_t ex = *(const int32_t *)(ids_host.data() + tok*ids->nb[1] + slot*ids->nb[0]);
                if (ex >= 0 && ex < n_expert) {
                    layer_counts[ex]++;
                    n_decisions++;
                }
            }
        }

        return true;
    }
};

static expert_profiler g_profiler;

static bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);
    return g_profiler.eval(t, ask);
}

static std::string ascii_bar(double frac, int width) {
    const int filled = std::max(0, std::min(width, (int) std::lround(frac * width)));
    return std::string(filled, '#') + std::string(width - filled, ' ');
}

static void report(const common_params & params) {
    const auto & counts = g_profiler.counts;

    if (counts.empty() || g_profiler.n_decisions == 0) {
        LOG_ERR("%s: no expert routing was captured - is this a MoE model?\n", __func__);
        return;
    }

    // flatten to (layer, expert, count) units; each unit is an independently placeable expert
    struct unit { int layer; int expert; int64_t count; };
    std::vector<unit> units;
    int64_t total = 0;
    int     n_expert = 0;
    for (const auto & [il, layer_counts] : counts) {
        n_expert = std::max(n_expert, (int) layer_counts.size());
        for (int ex = 0; ex < (int) layer_counts.size(); ++ex) {
            units.push_back({ il, ex, layer_counts[ex] });
            total += layer_counts[ex];
        }
    }

    std::sort(units.begin(), units.end(), [](const unit & a, const unit & b) {
        return a.count > b.count;
    });

    int64_t n_unused = 0;
    for (const auto & u : units) {
        if (u.count == 0) {
            n_unused++;
        }
    }

    LOG("\n");
    LOG("==== expert usage profile ====\n");
    LOG("moe layers profiled : %d\n", (int) counts.size());
    LOG("experts per layer   : %d\n", n_expert);
    LOG("expert units total  : %d  (layer x expert, independently placeable)\n", (int) units.size());
    LOG("routing decisions   : %lld  (token x experts-per-token)\n", (long long) total);
    LOG("unused expert units : %lld  (%.1f%%)\n", (long long) n_unused, 100.0 * n_unused / units.size());

    // cumulative coverage: if the hottest N% of units live in VRAM, what % of activations hit VRAM?
    LOG("\n-- VRAM coverage if hottest expert units are pinned --\n");
    LOG("%-12s %-12s\n", "% units", "% activations");
    int64_t running = 0;
    size_t  idx     = 0;
    for (int pct = 10; pct <= 100; pct += 10) {
        const size_t target = (size_t) std::llround(units.size() * (pct / 100.0));
        for (; idx < target && idx < units.size(); ++idx) {
            running += units[idx].count;
        }
        LOG("%-12d %-12.1f\n", pct, 100.0 * running / total);
    }

    // histogram of expert-unit hotness (share of total activations per unit)
    LOG("\n-- expert hotness histogram (share of all activations per unit) --\n");
    const int n_bins = 10;
    std::vector<int> bins(n_bins, 0);
    const double per_unit_mean = 1.0 / units.size(); // share if perfectly uniform
    const double bin_hi = 4.0 * per_unit_mean;       // bucket up to 4x the uniform share
    for (const auto & u : units) {
        const double share = (double) u.count / total;
        int b = (int) (share / bin_hi * n_bins);
        b = std::max(0, std::min(n_bins - 1, b));
        bins[b]++;
    }
    const int bin_max = std::max(1, *std::max_element(bins.begin(), bins.end()));
    for (int b = 0; b < n_bins; ++b) {
        const double lo = b * bin_hi / n_bins;
        const double hi = (b + 1) * bin_hi / n_bins;
        LOG("[%5.2fx-%5.2fx mean] %6d %s\n",
            lo / per_unit_mean, hi / per_unit_mean, bins[b],
            ascii_bar((double) bins[b] / bin_max, 40).c_str());
    }

    // hottest units
    const int n_top = (int) std::min<size_t>(units.size(), 20);
    LOG("\n-- hottest %d expert units --\n", n_top);
    LOG("%-8s %-8s %-12s %-10s\n", "layer", "expert", "count", "% of all");
    for (int i = 0; i < n_top; ++i) {
        LOG("%-8d %-8d %-12lld %-10.3f\n",
            units[i].layer, units[i].expert, (long long) units[i].count,
            100.0 * units[i].count / total);
    }

    // per-layer summary: how skewed is each layer
    LOG("\n-- per-layer summary --\n");
    LOG("%-8s %-10s %-10s %-10s %-14s\n", "layer", "used", "unused", "max share", "90%% in N exp");
    for (const auto & [il, layer_counts] : counts) {
        std::vector<int64_t> sorted(layer_counts.begin(), layer_counts.end());
        std::sort(sorted.begin(), sorted.end(), std::greater<int64_t>());

        int64_t layer_total = 0;
        int     used        = 0;
        for (int64_t c : sorted) {
            layer_total += c;
            if (c > 0) {
                used++;
            }
        }
        if (layer_total == 0) {
            continue;
        }

        int     n_cover = 0;
        int64_t acc     = 0;
        for (int64_t c : sorted) {
            acc += c;
            n_cover++;
            if (acc >= 0.9 * layer_total) {
                break;
            }
        }

        LOG("%-8d %-10d %-10d %-10.3f %-14d\n",
            il, used, (int) sorted.size() - used,
            (double) sorted[0] / layer_total, n_cover);
    }

    // full dump for offline plotting / placement planning
    if (!params.out_file.empty()) {
        std::ofstream f(params.out_file);
        if (f) {
            f << "layer,expert,count,frac_of_layer\n";
            for (const auto & [il, layer_counts] : counts) {
                int64_t layer_total = 0;
                for (int64_t c : layer_counts) {
                    layer_total += c;
                }
                for (int ex = 0; ex < (int) layer_counts.size(); ++ex) {
                    const double frac = layer_total ? (double) layer_counts[ex] / layer_total : 0.0;
                    f << il << "," << ex << "," << layer_counts[ex] << "," << frac << "\n";
                }
            }
            LOG("\nwrote per-expert counts to '%s'\n", params.out_file.c_str());
        } else {
            LOG_ERR("%s: failed to open '%s' for writing\n", __func__, params.out_file.c_str());
        }
    }
}

int main(int argc, char ** argv) {
    common_params params;

    params.out_file = "expert-usage.csv";
    params.n_ctx    = 4096;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_EXPERT_PROFILER, print_usage)) {
        return 1;
    }

    if (params.prompt.empty()) {
        LOG_ERR("%s: provide a prompt with -p or -f\n", __func__);
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = eval_callback;
    params.cb_eval_user_data = nullptr;
    params.warmup            = false;

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to load model\n", __func__);
        return 1;
    }

    LOG_INF("\n%s\n", common_params_get_system_info(params).c_str());

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true, true);
    if (tokens.empty()) {
        LOG_ERR("%s: prompt tokenized to zero tokens\n", __func__);
        return 1;
    }
    LOG_INF("%s: prompt is %d tokens\n", __func__, (int) tokens.size());

    const int n_ctx   = llama_n_ctx(ctx);
    const int n_batch = params.n_batch;

    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    // process the prompt in independent context-sized windows; routing stats only
    // depend on the forward pass, so windows do not need to share a KV cache
    for (size_t win = 0; win < tokens.size(); win += n_ctx) {
        const size_t win_end = std::min(tokens.size(), win + (size_t) n_ctx);
        llama_memory_clear(llama_get_memory(ctx), true);

        for (size_t off = win; off < win_end; off += n_batch) {
            const size_t end = std::min(win_end, off + (size_t) n_batch);
            common_batch_clear(batch);
            for (size_t i = off; i < end; ++i) {
                common_batch_add(batch, tokens[i], (llama_pos) (i - win), { 0 }, i + 1 == end);
            }
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("%s: llama_decode failed at token %zu\n", __func__, off);
                llama_batch_free(batch);
                return 1;
            }
            LOG_INF("%s: processed %zu / %zu tokens\n", __func__, end, tokens.size());
        }
    }

    llama_batch_free(batch);

    report(params);

    LOG("\n");
    llama_perf_context_print(ctx);
    llama_backend_free();

    return 0;
}
