#pragma once
#include <algorithm>
#include <limits>
#include <vector>

#include <hnswlib/hnswlib.h>

namespace vss {

typedef float (*SimFunc)(const float*, int, const float*, int, int);

inline float maxsim(const float* seq1, int len1, const float* seq2, int len2, int dim) {
    hnswlib::InnerProductSpace space(dim);
    hnswlib::DISTFUNC<float> dist_func = space.get_dist_func();
    void* dist_func_param = space.get_dist_func_param();

    float sum = 0.0f;
    const float* v1 = seq1;
    for (int i = 0; i < len1; i++, v1 += dim) {
        float sim = std::numeric_limits<float>::infinity();
        const float* v2 = seq2;
        for (int j = 0; j < len2; j++, v2 += dim) {
            sim = std::min(sim, dist_func(v1, v2, dist_func_param));
        }
        sum += sim;
    }
    return sum;
}

inline float dtw(const float* seq1, int len1, const float* seq2, int len2, int dim) {
    hnswlib::L2Space space(dim);
    hnswlib::DISTFUNC<float> dist_func = space.get_dist_func();
    void* dist_func_param = space.get_dist_func_param();

    const float INF = std::numeric_limits<float>::infinity();
    std::vector<float> pre(len2 + 1, INF), cur(len2 + 1, INF);
    pre[0] = 0;

    const float* v1 = seq1;
    for (int i = 1; i <= len1; i++, v1 += dim) {
        cur[0] = INF;
        const float* v2 = seq2;
        for (int j = 1; j <= len2; j++, v2 += dim) {
            float cost = dist_func(v1, v2, dist_func_param);
            cur[j] = cost + std::min({pre[j], cur[j - 1], pre[j - 1]});
        }
        std::swap(pre, cur);
    }
    return pre[len2];
}

} // namespace vss