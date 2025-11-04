#pragma once
#include <queue>

#include "dataset.h"
#include "metric.h"

namespace vss {

class VSSIndex {
public:
    int dim;
    SimMetric sim_metric;
    float (*sim_func)(const float*, int, const float*, int, int);

    VSSIndex(int dim, SimMetric sim_metric) : dim(dim), sim_metric(sim_metric) {
        if (sim_metric == MAXSIM) {
            sim_func = maxsim;
        } else if (sim_metric == DTW) {
            sim_func = dtw;
        } else if (sim_metric == SDTW) {
            sim_func = sdtw;
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

    long metric_cand_gen_time;
    long metric_rerank_time;

    virtual void build_vectors(const float* data, int size) = 0;
    virtual std::unordered_set<int> search_candidates(const float* q_data, int q_len, int q_k) = 0;

    RerankIndex(int dim, SimMetric sim_metric) : VSSIndex(dim, sim_metric) {}

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
        auto begin = std::chrono::high_resolution_clock::now();
        auto candidates = search_candidates(q_data, q_len, ef);
        auto mid = std::chrono::high_resolution_clock::now();

        std::priority_queue<std::pair<float, int>> result;
        for (int id : candidates) {
            float dist = sim_func(q_data, q_len, seq_data[id], seq_len[id], dim);
            result.emplace(dist, id);
            if (result.size() > k) {
                result.pop();
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        metric_cand_gen_time += std::chrono::duration_cast<std::chrono::microseconds>(mid - begin).count();
        metric_rerank_time += std::chrono::duration_cast<std::chrono::microseconds>(end - mid).count();

        return result;
    }

    std::vector<std::pair<std::string, long>> get_metrics() override {
        return {
            {"cand_gen_time", metric_cand_gen_time},
            {"rerank_time", metric_rerank_time},
        };
    }

    void reset_metrics() override {
        metric_cand_gen_time = 0;
        metric_rerank_time = 0;
    }
};

} // namespace vss
