#pragma once
#include "dataset.h"
#include "index.h"
#include "util.h"

namespace vss {

class BruteForceIndex : public RerankIndex {
public:
    BruteForceIndex(int dim, std::string sim_metric) : RerankIndex(dim, sim_metric) {}

    void build_vectors(const float* data, int size) override {}

    std::unordered_set<int> search_candidates(const float* q_data, int q_len, int q_k) override {
        std::unordered_set<int> candidates;
        for (int i = 0; i < seq_num; i++) {
            candidates.insert(i);
        }
        return candidates;
    }
};

} // namespace vss
