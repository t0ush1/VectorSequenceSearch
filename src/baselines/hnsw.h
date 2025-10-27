#pragma once

#include <hnswlib/hnswlib.h>

#include "dataset.h"
#include "index.h"

namespace vss {

class HNSWIndex : public RerankIndex {
public:
    int M;
    int ef_construction;
    hnswlib::SpaceInterface<float>* space;
    hnswlib::HierarchicalNSW<float>* hnsw;

    HNSWIndex(int dim, std::string sim_metric, int M, int ef_construction)
        : RerankIndex(dim, sim_metric), M(M), ef_construction(ef_construction) {}

    ~HNSWIndex() {
        delete hnsw;
        delete space;
    }

    void build_vectors(const float* data, int size) override {
        if (sim_metric == "maxsim") {
            space = new hnswlib::InnerProductSpace(dim);
        } else if (sim_metric == "l2") {
            space = new hnswlib::L2Space(dim);
        }

        hnsw = new hnswlib::HierarchicalNSW<float>(space, size, M, ef_construction);

        const float* vec = data;
        for (size_t i = 0; i < size; i++, vec += dim) {
            hnsw->addPoint(vec, i);
        }
    }

    std::unordered_set<int> search_candidates(const float* q_data, int q_len, int q_k) override {
        hnsw->ef_ = q_k;
        std::unordered_set<int> candidates;

        const float* q_vec = q_data;
        for (int i = 0; i < q_len; i++, q_vec += dim) {
            auto res = hnsw->searchKnn(q_vec, q_k);
            while (!res.empty()) {
                auto result = res.top();
                res.pop();
                candidates.insert(vec_to_seq[result.second]);
            }
        }

        return candidates;
    }
};

} // namespace vss
