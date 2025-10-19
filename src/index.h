#pragma once
#include <queue>

#include "dataset.h"
#include "util.h"

namespace vss {

class VSSIndex {
public:
    int dim;

    VSSIndex(int dim) : dim(dim) {}

    virtual void build(const VSSDataset* base_dataset) = 0;
    virtual std::priority_queue<std::pair<float, int>> search(const float* q_data, int q_len, int k) = 0;
};

class RerankIndex : public VSSIndex {
public:
    int base_num;
    std::vector<const float*> base_data;
    std::vector<int> base_length;

    int base_vec_num;
    std::vector<int> label_to_base;

    RerankIndex(int dim) : VSSIndex(dim) {}

    void build(const VSSDataset* base_dataset) override {
        base_num = base_dataset->seq_num;
        base_data = base_dataset->seq_datas;
        base_length = base_dataset->seq_lengths;
        base_vec_num = base_dataset->size;

        label_to_base.resize(base_vec_num);
        int label = 0;
        for (int i = 0; i < base_num; i++) {
            for (int j = 0; j < base_length[i]; j++) {
                label_to_base[label++] = i;
            }
        }

        build_index();
    }

    std::priority_queue<std::pair<float, int>> search(const float* q_data, int q_len, int k) override {
        int k_ = k / 2;
        std::unordered_set<int> candidates;
        while (candidates.size() < k && k_ <= base_vec_num) {
            search_index(candidates, q_data, q_len, k_);
            k_ *= 2;
        }

        std::priority_queue<std::pair<float, int>> result;
        for (int id : candidates) {
            float dist = dtw(q_data, q_len, base_data[id], base_length[id], dim);
            result.emplace(dist, id);
            if (result.size() > k) {
                result.pop();
            }
        }
        return result;
    }

private:
    virtual void build_index() = 0;

    virtual void search_index(std::unordered_set<int>& candidates, const float* q_data, int q_len, int k) = 0;

    // TODO 持久化
};

} // namespace vss
