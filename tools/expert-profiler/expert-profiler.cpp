#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
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
    LOG("\n    %s -m model.gguf -f prompts.txt [-c 4096] [-n 256] [-o expert-usage.csv]\n", argv[0]);
    LOG("\n    -f takes one prompt per line; '\\n' within a line becomes a newline\n");
    LOG("\n");
}

// per-layer expert selection counts, keyed by layer index
struct expert_profiler {
    std::mutex mutex;

    std::map<int, std::vector<int64_t>> counts; // layer -> [n_expert] selection counts
    int64_t n_decisions         = 0;            // total (token, slot) routing decisions seen
    int64_t n_decisions_prefill = 0;
    int64_t n_decisions_decode  = 0;
    bool    decode_phase        = false;        // set by the driver around generation

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

        int64_t local = 0;
        for (int64_t tok = 0; tok < ids->ne[1]; ++tok) {
            for (int64_t slot = 0; slot < ids->ne[0]; ++slot) {
                const int32_t ex = *(const int32_t *)(ids_host.data() + tok*ids->nb[1] + slot*ids->nb[0]);
                if (ex >= 0 && ex < n_expert) {
                    layer_counts[ex]++;
                    local++;
                }
            }
        }
        n_decisions += local;
        (decode_phase ? n_decisions_decode : n_decisions_prefill) += local;

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

struct run_totals {
    int     n_prompts = 0;
    int64_t n_prefill = 0; // prompt tokens fed
    int64_t n_decode  = 0; // tokens generated
};

static void report(const common_params & params, const run_totals & rt) {
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
    LOG("prompts processed   : %d\n", rt.n_prompts);
    LOG("prefill tokens      : %lld\n", (long long) rt.n_prefill);
    LOG("decode tokens gen   : %lld\n", (long long) rt.n_decode);
    LOG("moe layers profiled : %d\n", (int) counts.size());
    LOG("experts per layer   : %d\n", n_expert);
    LOG("expert units total  : %d  (layer x expert, independently placeable)\n", (int) units.size());
    LOG("routing decisions   : %lld  (prefill: %lld, decode: %lld)\n",
        (long long) total, (long long) g_profiler.n_decisions_prefill, (long long) g_profiler.n_decisions_decode);
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

// prefill a prompt, then autoregressively generate up to n_gen tokens, profiling both phases
static bool process_prompt(llama_context * ctx, const llama_vocab * vocab, common_sampler * smpl,
                           const std::vector<llama_token> & prompt, llama_batch & batch,
                           int n_ctx, int n_batch, int n_gen, run_totals & rt) {
    llama_memory_clear(llama_get_memory(ctx), true);
    common_sampler_reset(smpl);

    int n_prompt = (int) prompt.size();
    if (n_prompt >= n_ctx) {
        LOG_WRN("%s: prompt of %d tokens exceeds context %d, truncating\n", __func__, n_prompt, n_ctx);
        n_prompt = n_ctx - 1;
    }

    // prefill
    g_profiler.decode_phase = false;
    int n_past   = 0;
    int last_idx = 0;
    for (int off = 0; off < n_prompt; off += n_batch) {
        const int end = std::min(n_prompt, off + n_batch);
        common_batch_clear(batch);
        for (int i = off; i < end; ++i) {
            const bool is_last = (i + 1 == n_prompt);
            common_batch_add(batch, prompt[i], n_past++, { 0 }, is_last);
            if (is_last) {
                last_idx = batch.n_tokens - 1;
            }
        }
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: decode failed during prefill\n", __func__);
            return false;
        }
    }
    rt.n_prefill += n_prompt;

    // decode (generation)
    g_profiler.decode_phase = true;
    int idx = last_idx;
    for (int g = 0; g < n_gen && n_past < n_ctx; ++g) {
        const llama_token tok = common_sampler_sample(smpl, ctx, idx);
        common_sampler_accept(smpl, tok, true);
        if (llama_vocab_is_eog(vocab, tok)) {
            break;
        }
        common_batch_clear(batch);
        common_batch_add(batch, tok, n_past++, { 0 }, true);
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: decode failed during generation\n", __func__);
            return false;
        }
        rt.n_decode++;
        idx = 0;
    }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;

    params.out_file  = "expert-usage.csv";
    params.n_ctx     = 4096;
    params.n_predict = 256;   // tokens to generate per prompt
    params.escape    = false; // keep file text raw; we split per line and unescape ourselves

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_EXPERT_PROFILER, print_usage)) {
        return 1;
    }

    if (params.prompt.empty()) {
        LOG_ERR("%s: provide a prompt with -p or -f\n", __func__);
        return 1;
    }

    // one prompt per line; literal "\n" inside a line becomes a real newline
    std::vector<std::string> prompts;
    for (std::string line : string_split<std::string>(params.prompt, '\n')) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.find_first_not_of(" \t") == std::string::npos) {
            continue;
        }
        string_process_escapes(line);
        prompts.push_back(std::move(line));
    }
    if (prompts.empty()) {
        LOG_ERR("%s: no non-empty prompts found\n", __func__);
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

    const llama_vocab * vocab = llama_model_get_vocab(model);

    LOG_INF("\n%s\n", common_params_get_system_info(params).c_str());

    const int n_ctx   = llama_n_ctx(ctx);
    const int n_batch = params.n_batch;
    const int n_gen   = params.n_predict > 0 ? params.n_predict : 256;

    common_sampler * smpl = common_sampler_init(model, params.sampling);
    llama_batch      batch = llama_batch_init(n_batch, 0, 1);

    run_totals rt;
    for (size_t p = 0; p < prompts.size(); ++p) {
        std::vector<llama_token> tokens = common_tokenize(ctx, prompts[p], true, true);
        if (tokens.empty()) {
            LOG_WRN("%s: prompt %zu tokenized to zero tokens, skipping\n", __func__, p);
            continue;
        }
        rt.n_prompts++;
        LOG_INF("%s: prompt %zu/%zu: %d tokens, generating up to %d\n",
                __func__, p + 1, prompts.size(), (int) tokens.size(), n_gen);
        if (!process_prompt(ctx, vocab, smpl, tokens, batch, n_ctx, n_batch, n_gen, rt)) {
            break;
        }
    }

    llama_batch_free(batch);
    common_sampler_free(smpl);

    report(params, rt);

    LOG("\n");
    llama_perf_context_print(ctx);
    llama_backend_free();

    return 0;
}
