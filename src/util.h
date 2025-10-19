#pragma once
#include <algorithm>
#include <limits>
#include <vector>

#include <hnswlib/hnswlib.h>

namespace vss {

inline float l2_sq_dist(const float* v1, const float* v2, int dim) {
    float dist = 0;
    for (int i = 0; i < dim; i++) {
        float diff = v1[i] - v2[i];
        dist += diff * diff;
    }
    return dist;
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