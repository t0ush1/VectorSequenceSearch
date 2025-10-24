#pragma once

#include "dataset.h"
#include "index.h"
#include "my_hnsw.h"

namespace vss {

class MyHNSWIndex : public RerankIndex {
public:
    int M;
    int ef_construction;
    hnswlib::SpaceInterface<float>* space;
    MyHNSW<float>* hnsw;

    MyHNSWIndex(int dim, std::string sim_metric, int M, int ef_construction)
        : RerankIndex(dim, sim_metric), M(M), ef_construction(ef_construction) {}

    ~MyHNSWIndex() {
        delete hnsw;
        delete space;
    }

    void build_vectors(const float* data, int size) override {
        if (sim_metric == "maxsim") {
            space = new hnswlib::InnerProductSpace(dim);
        } else if (sim_metric == "l2") {
            space = new hnswlib::L2Space(dim);
        }

        hnsw = new MyHNSW<float>(space, size, M, ef_construction);

        const float* vec = data;
        for (size_t i = 0; i < size; i++, vec += dim) {
            hnsw->add_point(vec, i);
        }
    }

    std::unordered_set<int> search_candidates(const float* q_data, int q_len, int q_k) override {
        hnsw->ef = q_k;
        std::unordered_set<int> candidates;

        const float* q_vec = q_data;
        for (int i = 0; i < q_len; i++, q_vec += dim) {
            auto res = hnsw->search_knn(q_vec, q_k);
            while (!res.empty()) {
                auto result = res.top();
                res.pop();
                candidates.insert(label_to_base[result.second]);
            }
        }

        return candidates;
    }

    long get_metric(std::string metric_name) override {
        if (metric_name == "hops") {
            return hnsw->metric_hops;
        } else if (metric_name == "dist_comps") {
            return hnsw->metric_distance_computations;
        } else {
            return 0;
        }
    }

    void reset_metric() override {
        hnsw->metric_distance_computations = 0;
        hnsw->metric_hops = 0;
    }
};

} // namespace vss
