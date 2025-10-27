#pragma once
#include <queue>

#include "dataset.h"
#include "util.h"

namespace vss {

class VSSIndex {
public:
    int dim;
    std::string sim_metric;
    SimFunc sim_func;

    VSSIndex(int dim, std::string sim_metric) : dim(dim), sim_metric(sim_metric) {
        if (sim_metric == "maxsim") {
            sim_func = vss::maxsim;
        } else if (sim_metric == "dtw") {
            sim_func = vss::dtw;
        } else {
            std::cerr << "Unknown similarity metric: " << sim_metric << std::endl;
            std::exit(-1);
        }
    }

    virtual void build(const VSSDataset* base_dataset) = 0;
    virtual std::priority_queue<std::pair<float, int>> search(const float* q_data, int q_len, int k, int ef) = 0;
    virtual std::vector<std::pair<std::string, long>> get_metrics() { return {}; };
    virtual void reset_metrics() {};
};

class RerankIndex : public VSSIndex {
public:
    int seq_num;
    std::vector<const float*> seq_data;
    std::vector<int> seq_len;

    std::vector<int> vec_to_seq;

    virtual void build_vectors(const float* data, int size) = 0;
    virtual std::unordered_set<int> search_candidates(const float* q_data, int q_len, int q_k) = 0;

    RerankIndex(int dim, std::string sim_metric) : VSSIndex(dim, sim_metric) {}

    void build(const VSSDataset* base_dataset) override {
        seq_num = base_dataset->seq_num;
        seq_data = base_dataset->seq_data;
        seq_len = base_dataset->seq_len;

        vec_to_seq.resize(base_dataset->size);
        int label = 0;
        for (int i = 0; i < seq_num; i++) {
            for (int j = 0; j < seq_len[i]; j++) {
                vec_to_seq[label++] = i;
            }
        }

        build_vectors(base_dataset->data, base_dataset->size);
    }

    std::priority_queue<std::pair<float, int>> search(const float* q_data, int q_len, int k, int ef) override {
        std::priority_queue<std::pair<float, int>> result;
        auto candidates = search_candidates(q_data, q_len, ef);
        for (int id : candidates) {
            float dist = sim_func(q_data, q_len, seq_data[id], seq_len[id], dim);
            result.emplace(dist, id);
            if (result.size() > k) {
                result.pop();
            }
        }
        return result;
    }
};

} // namespace vss
