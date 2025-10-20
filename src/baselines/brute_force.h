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

    BruteForceIndex(int dim) : VSSIndex(dim) {}

    void build(const VSSDataset* base_dataset) override {
        // TODO 复制数据
        base_num = base_dataset->seq_num;
        base_data = base_dataset->seq_datas;
        base_length = base_dataset->seq_lengths;
    }

    std::priority_queue<std::pair<float, int>> search(const float* q_data, int q_len, int k, int ef) override {
        std::priority_queue<std::pair<float, int>> result;
        for (int i = 0; i < base_num; i++) {
            float dist = dtw(q_data, q_len, base_data[i], base_length[i], dim);
            result.emplace(dist, i);
            if (result.size() > k) {
                result.pop();
            }   
        }
        return result;
    }
};

} // namespace vss
