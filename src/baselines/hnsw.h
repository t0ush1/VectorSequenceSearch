#pragma once

#include <hnswlib/hnswlib.h>

#include "dataset.h"
#include "index.h"

namespace vss {

class HNSWIndex : public RerankIndex {
public:
    int M;
    int ef_construction;
    hnswlib::L2Space* space;
    hnswlib::HierarchicalNSW<float>* hnsw;

    HNSWIndex(int dim, int M, int ef_construction) : RerankIndex(dim), M(M), ef_construction(ef_construction) {}

    ~HNSWIndex() {
        delete hnsw;
        delete space;
    }

    void build_index() override {
        space = new hnswlib::L2Space(dim);
        hnsw = new hnswlib::HierarchicalNSW<float>(space, base_vec_num, M, ef_construction);

        hnswlib::labeltype label = 0;
        for (int i = 0; i < base_num; i++) {
            const float* vec = base_data[i];
            for (int j = 0; j < base_length[i]; j++) {
                hnsw->addPoint(vec, label);
                vec += dim;
                label++;
            }
        }
    }

    void search_index(std::unordered_set<int>& candidates, const float* q_data, int q_len, int k, int ef) override {
        hnsw->ef_ = ef;

        const float* q_vec = q_data;
        for (int i = 0; i < q_len; i++, q_vec += dim) {
            auto res = hnsw->searchKnn(q_vec, k);
            while (!res.empty()) {
                auto [dist, label] = res.top();
                res.pop();
                candidates.insert(label_to_base[label]);
            }
        }
    }
};

} // namespace vss
