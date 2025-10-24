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
    int base_num;
    std::vector<const float*> base_data;
    std::vector<int> base_length;

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
        base_num = base_dataset->seq_num;
        base_data = base_dataset->seq_datas;
        base_length = base_dataset->seq_lengths;

        if (sim_metric == "maxsim") {
            space = new hnswlib::InnerProductSpace(dim);
        } else if (sim_metric == "dtw") {
            space = new hnswlib::L2Space(dim);
        }

        sequence_hnsw = new SequenceHNSW<dist_t, slabel_t>(space, base_dataset->size, M, ef_construction);

        std::vector<std::vector<label_t>> sequences;
        label_t label = 0;
        for (int i = 0; i < base_num; i++) {
            std::vector<label_t> seq;
            const float* vec = base_data[i];
            for (int j = 0; j < base_length[i]; j++) {
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
            float dist = dtw(q_data, q_len, base_data[id], base_length[id], dim);
            result.emplace(dist, id);
            if (result.size() > k) {
                result.pop();
            }
        }
        return result;
    }

    long get_metric(std::string metric_name) override {
        if (metric_name == "hops") {
            return sequence_hnsw->metric_hops;
        } else if (metric_name == "dist_comps") {
            return sequence_hnsw->metric_distance_computations;
        } else {
            return 0;
        }
    }

    void reset_metric() override {
        sequence_hnsw->metric_distance_computations = 0;
        sequence_hnsw->metric_hops = 0;
    }
};

} // namespace vss
