#pragma once

#include "dataset.h"
#include "index.h"
#include "my_hnsw.h"

namespace vss {

class MyHNSWIndex : public RerankIndex {
public:
    int M;
    int ef_construction;
    hnswlib::L2Space* space;
    MyHNSW<float>* hnsw;

    MyHNSWIndex(int dim, int M, int ef_construction) : RerankIndex(dim), M(M), ef_construction(ef_construction) {}

    ~MyHNSWIndex() {
        delete hnsw;
        delete space;
    }

    void build_index() override {
        space = new hnswlib::L2Space(dim);
        hnsw = new MyHNSW<float>(space, base_vec_num, M, ef_construction);

        hnswlib::labeltype label = 0;
        for (int i = 0; i < base_num; i++) {
            const float* vec = base_data[i];
            for (int j = 0; j < base_length[i]; j++) {
                hnsw->add_point(vec, label);
                vec += dim;
                label++;
            }
        }
    }

    void search_index(std::unordered_set<int>& candidates, const float* q_data, int q_len, int k, int ef) override {
        hnsw->ef = ef;

        const float* q_vec = q_data;
        for (int i = 0; i < q_len; i++, q_vec += dim) {
            auto res = hnsw->search_knn(q_vec, k);
            while (!res.empty()) {
                auto [dist, label] = res.top();
                res.pop();
                candidates.insert(label_to_base[label]);
            }
        }
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
