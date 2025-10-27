#pragma once

#include <hnswlib/hnswlib.h>

#include "dataset.h"
#include "index.h"
#include "sequence_hnsw.h"
#include "util.h"

namespace vss {

template<typename dist_t, typename slabel_t>
class SequenceGraphIndex : public VSSIndex {
public:
    int seq_num;
    std::vector<const float*> seq_data;
    std::vector<int> seq_len;

    int M;
    int ef_construction;
    hnswlib::SpaceInterface<dist_t>* space;
    SequenceHNSW<dist_t, slabel_t>* sequence_hnsw;

    SequenceGraphIndex(int dim, std::string sim_metric, int M, int ef_construction)
        : VSSIndex(dim, sim_metric), M(M), ef_construction(ef_construction) {}

    ~SequenceGraphIndex() {
        delete sequence_hnsw;
        delete space;
    }

    void build(const VSSDataset* base_dataset) override {
        seq_num = base_dataset->seq_num;
        seq_data = base_dataset->seq_data;
        seq_len = base_dataset->seq_len;

        if (sim_metric == "maxsim") {
            space = new hnswlib::InnerProductSpace(dim);
        } else if (sim_metric == "dtw") {
            space = new hnswlib::L2Space(dim);
        }

        sequence_hnsw = new SequenceHNSW<dist_t, slabel_t>(space, base_dataset->size, M, ef_construction);

        std::vector<std::vector<label_t>> sequences;
        label_t label = 0;
        for (int i = 0; i < seq_num; i++) {
            std::vector<label_t> seq;
            const float* vec = seq_data[i];
            for (int j = 0; j < seq_len[i]; j++) {
                sequence_hnsw->add_point(vec, label, i, j);
                seq.push_back(label);
                vec += dim;
                label++;
            }
            sequences.push_back(seq);
        }
        for (const auto& seq : sequences) {
            sequence_hnsw->add_successors(seq);
        }
    }

    std::priority_queue<std::pair<float, int>> search(const float* q_data, int q_len, int k, int ef) override {
        sequence_hnsw->ef = ef;

        std::priority_queue<std::pair<float, int>> result;
        auto candidates = sequence_hnsw->search_candidates(q_data, q_len);
        for (slabel_t id : candidates) {
            float dist = dtw(q_data, q_len, seq_data[id], seq_len[id], dim);
            result.emplace(dist, id);
            if (result.size() > k) {
                result.pop();
            }
        }
        return result;
    }

    std::vector<std::pair<std::string, long>> get_metrics() override {
        return {
            {"hops", sequence_hnsw->metric_hops},
            {"dist_comps", sequence_hnsw->metric_distance_computations},
        };
    }

    void reset_metrics() override {
        sequence_hnsw->metric_distance_computations = 0;
        sequence_hnsw->metric_hops = 0;
    }
};

} // namespace vss
