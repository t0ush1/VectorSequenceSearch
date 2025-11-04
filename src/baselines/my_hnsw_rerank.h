#pragma once

#include "index.h"
#include "my_hnsw.h"

namespace vss {

class MyHNSWIndex : public RerankIndex {
public:
    int M;
    int ef_construction;
    hnswlib::SpaceInterface<float>* space;
    MyHNSW<float>* hnsw;

    MyHNSWIndex(int dim, SimMetric sim_metric, int M, int ef_construction)
        : RerankIndex(dim, sim_metric), M(M), ef_construction(ef_construction) {}

    ~MyHNSWIndex() {
        delete hnsw;
        delete space;
    }

    void build_vectors(const float* data, int size) override {
        if (sim_metric == MAXSIM) {
            space = new hnswlib::InnerProductSpace(dim);
        } else {
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
                candidates.insert(vec_to_seq[result.second]);
            }
        }

        return candidates;
    }

    std::vector<std::pair<std::string, long>> get_metrics() override {
        return {
            {"hops", hnsw->metric_hops},
            {"dist_comps", hnsw->metric_distance_computations},
            {"cand_gen_time", metric_cand_gen_time},
            {"rerank_time", metric_rerank_time},
        };
    }

    void reset_metrics() override {
        hnsw->metric_distance_computations = 0;
        hnsw->metric_hops = 0;
        metric_cand_gen_time = 0;
        metric_rerank_time = 0;
    }
};

} // namespace vss
