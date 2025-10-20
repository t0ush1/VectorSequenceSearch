#pragma once
#include "dataset.h"
#include "index.h"
#include "util.h"

namespace vss {

class BruteForceIndex : public VSSIndex {
public:
    int base_num;
    std::vector<const float*> base_data;
    std::vector<int> base_length;

    long metric_distance_computations;

    BruteForceIndex(int dim) : VSSIndex(dim) {}

    void build(const VSSDataset* base_dataset) override {
        base_num = base_dataset->seq_num;
        base_data = base_dataset->seq_datas;
        base_length = base_dataset->seq_lengths;
    }

    std::priority_queue<std::pair<float, int>> search(const float* q_data, int q_len, int k, int ef) override {
        std::priority_queue<std::pair<float, int>> result;
        for (int i = 0; i < base_num; i++) {
            metric_distance_computations += base_length[i] * q_len * dim;

            float dist = dtw(q_data, q_len, base_data[i], base_length[i], dim);
            result.emplace(dist, i);
            if (result.size() > k) {
                result.pop();
            }
        }
        return result;
    }

    long get_metric(std::string metric_name) override {
        if (metric_name == "dist_comps") {
            return metric_distance_computations;
        }
        return 0;
    }

    void reset_metric() override {
        metric_distance_computations = 0;
    }
};

} // namespace vss
