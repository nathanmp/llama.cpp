// Regression test for GGML_TENSOR_FLAG_SPLIT: marking a node with the flag must force the backend
// scheduler to start a new split at that node *without changing the computed result*. Uses a single
// CPU backend, so the only difference between the two runs is split granularity.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <vector>

static std::vector<float> run(bool mark_boundary, int & n_splits_out) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

    const int n = 64;
    struct ggml_init_params p = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_name(a, "a");
    ggml_set_input(a);

    ggml_tensor * b = ggml_scale(ctx, a, 2.0f); // 2a
    ggml_tensor * c = ggml_add  (ctx, b, b);    // 4a   <- boundary marked here
    ggml_tensor * d = ggml_scale(ctx, c, 0.5f); // 2a
    ggml_tensor * e = ggml_add  (ctx, d, a);    // 3a
    ggml_set_output(e);

    if (mark_boundary) {
        c->flags |= GGML_TENSOR_FLAG_SPLIT;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, e);

    ggml_backend_sched_t sched = ggml_backend_sched_new(&backend, &buft, 1, ggml_graph_size(gf), false, false);
    ggml_backend_sched_reserve(sched, gf);
    ggml_backend_sched_alloc_graph(sched, gf);

    std::vector<float> in(n);
    for (int i = 0; i < n; i++) {
        in[i] = (float) i - 7.5f;
    }
    ggml_backend_tensor_set(a, in.data(), 0, n * sizeof(float));

    ggml_backend_sched_graph_compute(sched, gf);
    n_splits_out = ggml_backend_sched_get_n_splits(sched);

    std::vector<float> out(n);
    ggml_backend_tensor_get(e, out.data(), 0, n * sizeof(float));

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return out;
}

int main() {
    int s0 = 0, s1 = 0;
    std::vector<float> r0 = run(false, s0);
    std::vector<float> r1 = run(true,  s1);

    double maxdiff = 0.0;
    for (size_t i = 0; i < r0.size(); i++) {
        maxdiff = std::max(maxdiff, (double) std::fabs(r0[i] - r1[i]));
    }

    printf("test-sched-split: splits without flag = %d, with flag = %d, max|diff| = %.3e\n", s0, s1, maxdiff);

    if (s1 <= s0) {
        printf("FAIL: GGML_TENSOR_FLAG_SPLIT did not force an extra split\n");
        return 1;
    }
    if (maxdiff != 0.0) {
        printf("FAIL: result changed when a split boundary was forced\n");
        return 1;
    }
    printf("OK\n");
    return 0;
}
