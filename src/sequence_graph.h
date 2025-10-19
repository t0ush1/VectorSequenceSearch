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
    int ef;
    hnswlib::SpaceInterface<dist_t>* space;
    SequenceHNSW<dist_t, slabel_t>* sequence_hnsw;

    SequenceGraphIndex(int dim, int M = 16, int ef_construction = 200, int ef = 10)
        : VSSIndex(dim), M(M), ef_construction(ef_construction), ef(ef) {}

    ~SequenceGraphIndex() {
        delete sequence_hnsw;
        delete space;
    }

    void build(const VSSDataset* base_dataset) override {
        base_num = base_dataset->seq_num;
        base_data = base_dataset->seq_datas;
        base_length = base_dataset->seq_lengths;

        space = new hnswlib::L2Space(dim);
        sequence_hnsw = new SequenceHNSW<dist_t, slabel_t>(space, base_dataset->size, M, ef_construction);
        sequence_hnsw->ef = ef;

        // std::vector<std::pair<int, int>> ij_pair;
        // for (int i = 0; i < base_num; i++) {
        //     for (int j = 0; j < base_length[i]; j++) {
        //         ij_pair.emplace_back(i, j);
        //     }
        // }

        // std::vector<int> rand_indices(ij_pair.size());
        // for (int i = 0; i < rand_indices.size(); i++) {
        //     rand_indices[i] = i;
        // }
        // std::shuffle(rand_indices.begin(), rand_indices.end(), std::default_random_engine(42));

        // for (int id : rand_indices) {
        //     auto [i, j] = ij_pair[id];
        //     sequence_hnsw->add_point(base_data[i] + j * dim, id, i, j);
        // }

        // label_t label = 0;
        // for (int i = 0; i < base_num; i++) {
        //     std::vector<label_t> seq;
        //     for (int j = 0; j < base_length[i]; j++) {
        //         seq.push_back(label++);
        //     }
        //     sequence_hnsw->add_successors(seq);
        // }

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

    std::priority_queue<std::pair<float, int>> search(const float* q_data, int q_len, int k) override {
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
};

} // namespace vss
