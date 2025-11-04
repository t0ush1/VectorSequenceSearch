#pragma once

#include <algorithm>
#include <chrono>
#include <random>

#include <hnswlib/hnswlib.h>

#include "index.h"
#include "seq_hnsw.h"

namespace vss {

struct Status {
    vid_t vid;
    lid_t q_lid;
    lid_t b_lid;
    float dist;

    // bool operator<(const Status& other) const { return dist < other.dist; }

    // bool operator>(const Status& other) const { return dist > other.dist; }

    bool operator<(const Status& other) const {
        return q_lid > other.q_lid || (q_lid == other.q_lid && dist < other.dist);
    }

    bool operator>(const Status& other) const {
        return q_lid < other.q_lid || (q_lid == other.q_lid && dist > other.dist);
    }
};

class VisitedStatus {
public:
    size_t vec_num;
    size_t max_len;
    uint8_t tag;
    uint8_t* mass;
    float* buffer;

    VisitedStatus(size_t vec_num) : vec_num(vec_num), max_len(0), tag(-1), mass(nullptr), buffer(nullptr) {}

    inline void reset(size_t q_len) {
        if (q_len > max_len) {
            mass = (uint8_t*)realloc(mass, sizeof(uint8_t) * vec_num * q_len);
            buffer = (float*)realloc(buffer, sizeof(float) * vec_num * q_len);
            memset(mass + max_len * vec_num, 0, sizeof(uint8_t) * vec_num * (q_len - max_len));
            max_len = q_len;
        }
        tag++;
        if (tag == 0) {
            tag = 1;
            memset(mass, 0, sizeof(uint8_t) * vec_num * q_len);
        }
    }

    inline void visit(const Status& status, float dist) {
        int index = status.q_lid * vec_num + status.vid;
        mass[index] = tag;
        buffer[index] = dist;
    }

    inline bool is_visited(const Status& status) const { return mass[status.q_lid * vec_num + status.vid] == tag; }

    ~VisitedStatus() {
        free(mass);
        free(buffer);
    }
};

template<bool enable_buffer, bool unlink_same_seq>
class SeqGraphIndex : public VSSIndex {
public:
    int seq_num;
    std::vector<const float*> seq_data;
    std::vector<int> seq_len;
    std::vector<int> seq_off;

    int vec_num;
    std::vector<vid_t> v2v;
    std::vector<sid_t> v2s;
    std::vector<lid_t> v2l;

    int M;
    int ef_construction;
    hnswlib::SpaceInterface<float>* space;
    SeqHNSW<float, enable_buffer, unlink_same_seq>* hnsw;

    VisitedStatus* visited_status;
    float (*sim_func_from_matrix)(const float* dist_matrix, int buffer_len, int q_len, int b_len);

    long metric_buffer_hit;
    long metric_buffer_tot;
    long metric_cand_gen_time;
    long metric_rerank_time;

    SeqGraphIndex(int dim, SimMetric sim_metric, int M, int ef_construction)
        : VSSIndex(dim, sim_metric), M(M), ef_construction(ef_construction) {
        assert(sim_metric == DTW || sim_metric == SDTW);
        if (sim_metric == DTW) {
            sim_func_from_matrix = dtw_from_matrix;
        } else if (sim_metric == SDTW) {
            sim_func_from_matrix = sdtw_from_matrix;
        }
    }

    ~SeqGraphIndex() {
        delete hnsw;
        delete space;
        delete visited_status;
    }

    void build(const VSSDataset* base_dataset) {
        seq_num = base_dataset->seq_num;
        seq_data = base_dataset->seq_data;
        seq_len = base_dataset->seq_len;
        seq_off.resize(seq_num);
        seq_off[0] = 0;
        for (int i = 1; i < seq_num; i++) {
            seq_off[i] = seq_off[i - 1] + seq_len[i - 1];
        }

        vec_num = base_dataset->size;
        v2s.resize(vec_num);
        v2l.resize(vec_num);

        std::vector<vid_t> vids(vec_num);
        vid_t vid = 0;
        for (sid_t i = 0; i < seq_num; i++) {
            for (lid_t j = 0; j < seq_len[i]; j++, vid++) {
                vids[vid] = vid;
                v2s[vid] = i;
                v2l[vid] = j;
            }
        }

        if (unlink_same_seq) {
            std::shuffle(vids.begin(), vids.end(), std::default_random_engine(100));
        }

        // TODO 如果要考虑内存连续性，加内外层id映射
        space = new hnswlib::L2Space(dim);
        hnsw = new SeqHNSW<float, enable_buffer, unlink_same_seq>(space, vec_num, M, ef_construction);
        hnsw->outer_index = this;
        for (int id : vids) {
            hnsw->add_point(seq_data[v2s[id]] + v2l[id] * dim, id);
        }

        visited_status = new VisitedStatus(vec_num);
    }

    std::priority_queue<Status> search_level_dp(id_t ep_id, const void* q_data, int q_len, int level) {
        visited_status->reset(q_len);
        std::priority_queue<Status, std::vector<Status>, std::less<Status>> top_candidates;
        std::priority_queue<Status, std::vector<Status>, std::greater<Status>> candidate_set;
        Status lower_bound = {ep_id, 0, 0, hnsw->fstdistfunc(q_data, hnsw->addr_data(ep_id), hnsw->dist_func_param)};
        top_candidates.emplace(lower_bound);
        candidate_set.emplace(lower_bound);
        visited_status->visit(lower_bound, lower_bound.dist);

        auto visit_status = [&](Status st) {
            if (visited_status->is_visited(st)) {
                return;
            }

            float dist = hnsw->fstdistfunc((char*)q_data + st.q_lid * hnsw->data_size, hnsw->addr_data(st.vid),
                                           hnsw->dist_func_param);
            st.dist += dist;
            visited_status->visit(st, dist);
            hnsw->metric_distance_computations++;

            if (top_candidates.size() < hnsw->ef || st < lower_bound) {
                if (st.q_lid == q_len - 1) {
                    top_candidates.emplace(st);
                    if (top_candidates.size() > hnsw->ef) {
                        top_candidates.pop();
                    }
                    lower_bound = top_candidates.top();
                } else {
                    candidate_set.emplace(st);
                }
            }
        };

        while (!candidate_set.empty()) {
            Status st = candidate_set.top();
            if (st > lower_bound && top_candidates.size() >= hnsw->ef) {
                break;
            }
            candidate_set.pop();

            hnsw->metric_hops++;

            linklist_t* ll = hnsw->addr_linklist(st.vid, level);
            int size = hnsw->get_ll_size(ll);
            vid_t* neighbors = hnsw->get_ll_neighbors(ll);

            visit_status({st.vid, st.q_lid + 1, st.b_lid, st.dist});
            if (st.b_lid < seq_len[v2s[st.vid]] - 1) {
                visit_status({st.vid + 1, st.q_lid, st.b_lid + 1, st.dist});
                visit_status({st.vid + 1, st.q_lid + 1, st.b_lid + 1, st.dist});
            }
            for (int i = 0; i < size; i++) {
                visit_status({neighbors[i], 0, v2l[neighbors[i]], 0});
            }
        }

        return top_candidates;
    }

    std::unordered_set<sid_t> generate_candidates(const float* q_data, int q_len, int ef) {
        hnsw->ef = ef;

        id_t ep_id = hnsw->template search_down_to_level<true>(hnsw->enterpoint, q_data, 0);
        auto top_candidates = search_level_dp(ep_id, q_data, q_len, 0);

        std::unordered_set<sid_t> unique;
        while (!top_candidates.empty()) {
            unique.insert(v2s[top_candidates.top().vid]);
            top_candidates.pop();
        }
        return unique;
    }

    void fill_dist_matrix(const float* q_data, int q_len, const float* b_data, int b_len, int b_off) {
        uint8_t* visited = visited_status->mass + b_off;
        float* buffer = visited_status->buffer + b_off;
        const float* vec1 = q_data;
        for (int i = 0; i < q_len; i++, vec1 += dim, visited += vec_num, buffer += vec_num) {
            const float* vec2 = b_data;
            for (int j = 0; j < b_len; j++, vec2 += dim) {
                if (visited[j] != visited_status->tag) {
                    buffer[j] = hnsw->fstdistfunc(vec1, vec2, hnsw->dist_func_param);
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
                fill_dist_matrix(q_data, q_len, seq_data[i], seq_len[i], seq_off[i]);
                dist = sim_func_from_matrix(visited_status->buffer + seq_off[i], vec_num, q_len, seq_len[i]);
            } else {
                dist = sim_func(q_data, q_len, seq_data[i], seq_len[i], dim);
            }

            result.emplace(dist, i);
            if (result.size() > k) {
                result.pop();
            }
        }
        return result;
    }

    std::priority_queue<std::pair<float, int>> search(const float* q_data, int q_len, int k, int ef) override {
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
