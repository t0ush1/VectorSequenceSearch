#pragma once

#include <algorithm>
#include <chrono>
#include <random>

#include <hnswlib/hnswlib.h>

#include "dataset.h"
#include "index.h"
#include "set_hnsw.h"

namespace vss {

template<bool enable_buffer, bool unlink_same_set>
class SetGraphIndex : public VSSIndex {
public:
    int set_num;
    std::vector<const float*> set_data;
    std::vector<int> set_len;
    std::vector<int> set_off;

    int vec_num;
    std::vector<sid_t> v2s;
    std::vector<lid_t> v2l;

    int M;
    int ef_construction;
    hnswlib::SpaceInterface<float>* space;
    SetHNSW<float, enable_buffer, unlink_same_set>* hnsw;

    hnswlib::DISTFUNC<float> dist_func;
    void* dist_func_param;
    float* dist_matrix;
    float* dist_buffer;
    float (*sim_func_from_matrix)(const float* dist_matrix, int buffer_len, int q_len, int b_len);

    long metric_buffer_hit;
    long metric_buffer_tot;
    long metric_cand_gen_time;
    long metric_rerank_time;

    SetGraphIndex(int dim, SimMetric sim_metric, int M, int ef_construction)
        : VSSIndex(dim, sim_metric), M(M), ef_construction(ef_construction) {}

    ~SetGraphIndex() {
        delete hnsw;
        delete space;
    }

    virtual void build(const VSSDataset* base_dataset) override {
        set_num = base_dataset->seq_num;
        set_data = base_dataset->seq_data;
        set_len = base_dataset->seq_len;
        set_off.resize(set_num);
        set_off[0] = 0;
        for (int i = 1; i < set_num; i++) {
            set_off[i] = set_off[i - 1] + set_len[i - 1];
        }

        vec_num = base_dataset->size;
        v2s.resize(vec_num);
        v2l.resize(vec_num);

        if (sim_metric == MAXSIM) {
            space = new hnswlib::InnerProductSpace(dim);
        } else {
            space = new hnswlib::L2Space(dim);
        }

        dist_func = space->get_dist_func();
        dist_func_param = space->get_dist_func_param();

        hnsw = new SetHNSW<float, enable_buffer, unlink_same_set>(space, vec_num, M, ef_construction);
        hnsw->outer_index = this;

        if (sim_metric == MAXSIM) {
            sim_func_from_matrix = maxsim_from_matrix;
        } else if (sim_metric == DTW) {
            sim_func_from_matrix = dtw_from_matrix;
        } else if (sim_metric == SDTW) {
            sim_func_from_matrix = sdtw_from_matrix;
        }

        std::vector<vid_t> vids(vec_num);
        vid_t vid = 0;
        for (sid_t i = 0; i < set_num; i++) {
            for (lid_t j = 0; j < set_len[i]; j++, vid++) {
                vids[vid] = vid;
                v2s[vid] = i;
                v2l[vid] = j;
            }
        }

        if (unlink_same_set) {
            std::shuffle(vids.begin(), vids.end(), std::default_random_engine(100));
        }

        // TODO 如果要考虑内存连续性，加一个id映射到dist_matrix的idx
        for (int id : vids) {
            hnsw->add_point(set_data[v2s[id]] + v2l[id] * dim, id);
        }
    }

    virtual std::unordered_set<sid_t> generate_candidates(const float* q_data, int q_len, int ef) {
        hnsw->ef = ef;

        std::unordered_set<sid_t> candidates;
        const float* q_vec = q_data;
        dist_buffer = dist_matrix;
        for (int i = 0; i < q_len; i++, q_vec += dim, dist_buffer += vec_num) {
            auto res = hnsw->search_knn(q_vec, ef);
            while (!res.empty()) {
                auto [dist, vid] = res.top();
                res.pop();
                candidates.insert(v2s[vid]);
            }
        }

        return candidates;
    }

    void fill_dist_matrix(const float* q_data, int q_len, const float* b_data, int b_len, int b_off) {
        float* buffer = dist_matrix + b_off;
        const float* vec1 = q_data;
        for (int i = 0; i < q_len; i++, vec1 += dim, buffer += vec_num) {
            const float* vec2 = b_data;
            for (int j = 0; j < b_len; j++, vec2 += dim) {
                if (buffer[j] == 0) {
                    buffer[j] = dist_func(vec1, vec2, dist_func_param);
                } else {
                    metric_buffer_hit++;
                }
            }
        }
        metric_buffer_tot += q_len * b_len;
    }

    std::priority_queue<std::pair<float, int>> rerank(const std::unordered_set<sid_t>& candidates, const float* q_data,
                                                      int q_len, int k) {
        std::priority_queue<std::pair<float, int>> result;
        for (sid_t i : candidates) {
            float dist;
            if (enable_buffer) {
                fill_dist_matrix(q_data, q_len, set_data[i], set_len[i], set_off[i]);
                dist = sim_func_from_matrix(dist_matrix + set_off[i], vec_num, q_len, set_len[i]);
            } else {
                dist = sim_func(q_data, q_len, set_data[i], set_len[i], dim);
            }

            result.emplace(dist, i);
            if (result.size() > k) {
                result.pop();
            }
        }
        return result;
    }

    std::priority_queue<std::pair<float, int>> search(const float* q_data, int q_len, int k, int ef) override {
        if (enable_buffer) {
            dist_matrix = (float*)realloc(dist_matrix, q_len * vec_num * sizeof(float));
            memset(dist_matrix, 0, q_len * vec_num * sizeof(float));
        }

        auto begin = std::chrono::high_resolution_clock::now();
        auto candidates = generate_candidates(q_data, q_len, ef);
        auto mid = std::chrono::high_resolution_clock::now();
        auto result = rerank(candidates, q_data, q_len, k);
        auto end = std::chrono::high_resolution_clock::now();
        metric_cand_gen_time += std::chrono::duration_cast<std::chrono::microseconds>(mid - begin).count();
        metric_rerank_time += std::chrono::duration_cast<std::chrono::microseconds>(end - mid).count();

        return result;
    }

    std::vector<std::pair<std::string, long>> get_metrics() override {
        return {
            {"hops", hnsw->metric_hops},
            {"dist_comps", hnsw->metric_distance_computations},
            {"buffer_hit", metric_buffer_hit},
            {"buffer_tot", metric_buffer_tot},
            {"cand_gen_time", metric_cand_gen_time},
            {"rerank_time", metric_rerank_time},
        };
    }

    void reset_metrics() override {
        hnsw->metric_distance_computations = 0;
        hnsw->metric_hops = 0;
        metric_buffer_hit = 0;
        metric_buffer_tot = 0;
        metric_cand_gen_time = 0;
        metric_rerank_time = 0;
    }
};

} // namespace vss
