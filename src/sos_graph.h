#pragma once

#include <bit>

#include <hnswlib/hnswlib.h>

#include "dataset.h"
#include "index.h"
#include "sos_hnsw.h"
#include "util.h"

namespace vss {

template<bool enable_buffer>
class SOSGraphIndex : public VSSIndex {
public:
    int sos_num;
    std::vector<const float*> sos_data;
    std::vector<int> sos_len;

    int vec_num;
    std::vector<int> v2s;
    std::vector<int> v2l;
    std::vector<int> s2v;

    int M;
    int ef_construction;
    hnswlib::SpaceInterface<float>* space;
    SOSHNSW<float, enable_buffer>* hnsw;

    std::vector<std::vector<float>> dist_matrix;
    hnswlib::DISTFUNC<float> dist_func;
    void* dist_func_param;

    long metric_buffer_hit;
    long metric_buffer_tot;

    SOSGraphIndex(int dim, std::string sim_metric, int M, int ef_construction)
        : VSSIndex(dim, sim_metric), M(M), ef_construction(ef_construction) {}

    ~SOSGraphIndex() {
        delete hnsw;
        delete space;
    }

    void build(const VSSDataset* base_dataset) override {
        sos_num = base_dataset->seq_num;
        sos_data = base_dataset->seq_data;
        sos_len = base_dataset->seq_len;

        vec_num = base_dataset->size;
        v2s.reserve(vec_num);
        v2l.reserve(vec_num);
        s2v.reserve(sos_num);

        if (sim_metric == "maxsim") {
            space = new hnswlib::InnerProductSpace(dim);
        } else if (sim_metric == "dtw") {
            space = new hnswlib::L2Space(dim);
        }

        dist_func = space->get_dist_func();
        dist_func_param = space->get_dist_func_param();

        hnsw = new SOSHNSW<float, enable_buffer>(space, vec_num, M, ef_construction);

        int vid = 0;
        const float* vec = sos_data[0];
        for (int i = 0; i < sos_num; i++) {
            s2v.push_back(vid);
            for (int j = 0; j < sos_len[i]; j++, vid++, vec += dim) {
                hnsw->add_point(vec);
                v2s.push_back(i);
                v2l.push_back(j);
            }
        }
    }

    inline float maxsim_from_matrix(const float* q_data, int q_len, const float* b_data, int b_len, int vid) {
        const float* vec1 = q_data;
        float sum = 0.0f;
        for (int i = 0; i < q_len; i++, vec1 += dim) {
            const float* vec2 = b_data;
            float sim = std::numeric_limits<float>::infinity();
            for (int j = 0; j < b_len; j++, vec2 += dim) {
                float dist = dist_matrix[i][vid + j];
                if (dist == 0) {
                    dist = dist_func(vec1, vec2, dist_func_param);
                } else {
                    metric_buffer_hit++;
                }
                sim = std::min(sim, dist);
            }
            sum += sim;
        }
        return sum;
    }

    std::priority_queue<std::pair<float, int>> search(const float* q_data, int q_len, int k, int ef) override {
        hnsw->ef = ef;

        if (enable_buffer) {
            if (q_len > dist_matrix.size()) {
                dist_matrix.resize(q_len, std::vector<float>(vec_num, 0));
            }
            for (int i = 0; i < q_len; i++) {
                std::fill(dist_matrix[i].begin(), dist_matrix[i].end(), 0);
            }
        }

        std::unordered_set<int> candidates;
        const float* q_vec = q_data;
        for (int i = 0; i < q_len; i++, q_vec += dim) {
            if (enable_buffer) {
                hnsw->dist_buffer = dist_matrix[i].data();
            }

            auto res = hnsw->search_knn(q_vec, ef);
            while (!res.empty()) {
                auto [dist, vid] = res.top();
                res.pop();
                candidates.insert(v2s[vid]);
            }
        }

        std::priority_queue<std::pair<float, int>> result;
        for (int i : candidates) {
            float dist;
            if (enable_buffer) {
                dist = maxsim_from_matrix(q_data, q_len, sos_data[i], sos_len[i], s2v[i]);
                metric_buffer_tot += q_len * sos_len[i];
            } else {
                dist = sim_func(q_data, q_len, sos_data[i], sos_len[i], dim);
            }

            result.emplace(dist, i);
            if (result.size() > k) {
                result.pop();
            }
        }

        return result;
    }

    std::vector<std::pair<std::string, long>> get_metrics() override {
        return {
            {"hops", hnsw->metric_hops},
            {"dist_comps", hnsw->metric_distance_computations},
            {"buffer_hit", metric_buffer_hit},
            {"buffer_tot", metric_buffer_tot},
        };
    }

    void reset_metrics() override {
        hnsw->metric_distance_computations = 0;
        hnsw->metric_hops = 0;
        metric_buffer_hit = 0;
        metric_buffer_tot = 0;
    }
};

} // namespace vss
