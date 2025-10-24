#pragma once

#include <hnswlib/hnswlib.h>

#include "dataset.h"
#include "index.h"
#include "sequence_hnsw.h"
#include "util.h"

namespace vss {

template<typename dist_t>
class SetGraphIndex : public VSSIndex {
public:
    int base_num;
    std::vector<const float*> base_data;
    std::vector<int> base_length;

    std::vector<int> label_to_base;

    int M;
    int ef_construction;
    hnswlib::SpaceInterface<dist_t>* space;
    hnswlib::HierarchicalNSW<dist_t>* hnsw;

    SetGraphIndex(int dim, std::string sim_metric, int M, int ef_construction)
        : VSSIndex(dim, sim_metric), M(M), ef_construction(ef_construction) {}

    ~SetGraphIndex() {
        delete hnsw;
        delete space;
    }

    void build(const VSSDataset* base_dataset) override {
        base_num = base_dataset->seq_num;
        base_data = base_dataset->seq_datas;
        base_length = base_dataset->seq_lengths;

        if (sim_metric == "maxsim") {
            space = new hnswlib::InnerProductSpace(dim);
        } else if (sim_metric == "dtw") {
            space = new hnswlib::L2Space(dim);
        }

        hnsw = new hnswlib::HierarchicalNSW<dist_t>(space, base_dataset->size, M, ef_construction);

        hnswlib::labeltype label = 0;
        for (int i = 0; i < base_num; i++) {
            const float* vec = base_data[i];
            for (int j = 0; j < base_length[i]; j++) {
                hnsw->addPoint(vec, label);
                label_to_base.push_back(i);
                vec += dim;
                label++;
            }
        }
    }

    std::priority_queue<std::pair<float, int>> search(const float* q_data, int q_len, int k, int ef) override {
        hnsw->ef_ = ef;

        std::vector<float> scores(base_num);

        const float* q_vec = q_data;
        for (int i = 0; i < q_len; i++, q_vec += dim) {
            auto res = hnsw->searchKnn(q_vec, ef);

            dist_t lower_bound = res.top().first;
            std::vector<float> dists(base_num, lower_bound);
            while (!res.empty()) {
                auto [dist, label] = res.top();
                res.pop();
                int base_id = label_to_base[label];
                dists[base_id] = std::min(dists[base_id], dist);
            }

            for (int j = 0; j < base_num; j++) {
                scores[j] += dists[j];
            }
        }

        std::priority_queue<std::pair<float, int>> result;
        for (int i = 0; i < base_num; i++) {
            result.emplace(scores[i], i);
            if (result.size() > k) {
                result.pop();
            }
        }
        return result;
    }

    long get_metric(std::string metric_name) override { return 0; }

    void reset_metric() override {}
};

} // namespace vss
