#pragma once
#include <chrono>
#include <filesystem>
#include <functional>
#include <unordered_map>
#include <unordered_set>

#include "baselines/brute_force.h"
#include "baselines/hnsw.h"
// #include "baselines/ivfpq.h"
#include "dataset.h"
#include "index.h"
#include "sequence_graph.h"

namespace vss {
class VSSRunner {
public:
    int dim;
    VSSDataset* base_dataset;
    VSSDataset* query_dataset;
    std::vector<std::unordered_set<int>> groundtruth;

    VSSIndex* index;

    VSSRunner(int dim, const char* data_dir, const char* index_type) : dim(dim) {
        fs::path data_path(data_dir);
        base_dataset = new VSSDataset(dim, data_path / "base.fvecs", data_path / "base.lens");
        query_dataset = new VSSDataset(dim, data_path / "query.fvecs", data_path / "query.lens");
        groundtruth = read_groundtruth(data_path / "groundtruth.ivecs");

        std::string name = index_type;
        if (name == "brute_force") {
            index = new BruteForceIndex(dim);
        } else if (name == "hnsw") {
            index = new HNSWIndex(dim, 16, 200, 10);
        // } else if (name == "ivfpq") {
        //     index = new IVFPQIndex(dim, 100, 8, 8, 10);
        } else if (name == "dtw") {
            // TODO 多ef测试框架
            index = new SequenceGraphIndex<float, int>(dim, 16, 200, 100);
        } else {
            std::cerr << "Unknown index: " << name << std::endl;
            std::exit(-1);
        }
    }

    ~VSSRunner() {
        delete base_dataset;
        delete query_dataset;
        delete index;
    }

    void run_build() {
        auto begin = std::chrono::high_resolution_clock::now();
        index->build(base_dataset);
        auto end = std::chrono::high_resolution_clock::now();
        size_t time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        std::cout << "Build Time: " << time << " us" << std::endl;
    }

    void run_search() {
        int k = groundtruth[0].size();
        int q_num = query_dataset->seq_num;
        int total = k * q_num;
        int hit = 0;

        size_t total_time = 0;

        for (int i = 0; i < q_num; i++) {
            auto [q_data, q_len] = query_dataset->get_sequence(i);

            auto begin = std::chrono::high_resolution_clock::now();
            auto result = index->search(q_data, q_len, k);
            auto end = std::chrono::high_resolution_clock::now();
            size_t time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            total_time += time;

            assert(result.size() <= k);
            while (result.size() > 0) {
                int id = result.top().second;
                result.pop();
                if (groundtruth[i].find(id) != groundtruth[i].end()) {
                    hit++;
                }
            }
        }

        std::cout << "Total Time: " << total_time << " us, Avg Time: " << total_time * 1.0 / q_num << " us"
                  << std::endl;
        std::cout << "Recall: " << hit << "/" << total << "=" << hit * 1.0 / total << std::endl;
    }
};

} // namespace vss