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
    struct QueryRecord {
        int ef;
        size_t time;
        int hit;
        int total;
        int q_num;
    };

    int dim;
    std::string data_dir;
    std::string index_name;

    VSSDataset* base_dataset;
    VSSDataset* query_dataset;
    std::vector<std::unordered_set<int>> groundtruth;

    VSSIndex* index;
    std::vector<int> efs;

    VSSRunner(int dim, std::string data_dir, std::string index_name)
        : dim(dim), data_dir(data_dir), index_name(index_name) {
        fs::path data_path = fs::path("../datasets") / data_dir;
        base_dataset = new VSSDataset(dim, data_path / "base.fvecs", data_path / "base.lens");
        query_dataset = new VSSDataset(dim, data_path / "query.fvecs", data_path / "query.lens");
        groundtruth = read_groundtruth(data_path / "groundtruth.ivecs");

        if (index_name == "brute_force") {
            index = new BruteForceIndex(dim);
            efs = {0};
        } else if (index_name == "hnsw") {
            index = new HNSWIndex(dim, 16, 200);
            efs = {10, 20, 50, 100, 200, 500, 1000};
            // } else if (index_name == "ivfpq") {
            //     index = new IVFPQIndex(dim, 100, 8, 8, 10);
        } else if (index_name == "dtw") {
            index = new SequenceGraphIndex<float, int>(dim, 16, 200);
            efs = {10, 20, 50, 100, 200, 500, 1000};
        } else {
            std::cerr << "Unknown index: " << index_name << std::endl;
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
        std::cout << std::endl;
    }

    void run_search() {
        std::vector<QueryRecord> records;
        int k = groundtruth[0].size();

        for (int ef : efs) {
            QueryRecord record = run_search_once(k, ef);
            records.push_back(record);
        }

        for (const auto& r : records) {
            std::cout << "EF: " << r.ef << std::endl;
            std::cout << "Tot Time: " << r.time << " us, Avg Time: " << r.time * 1.0 / r.q_num << " us" << std::endl;
            std::cout << "Recall: " << r.hit << "/" << r.total << "=" << r.hit * 1.0 / r.total << std::endl;
            std::cout << std::endl;
        }

        save_records(records);
    }

    QueryRecord run_search_once(int k, int ef) {
        size_t time = 0;
        int hit = 0;
        int total = 0;
        int q_num = 0;

        for (int i = 0; i < query_dataset->seq_num; i++) {
            auto [q_data, q_len] = query_dataset->get_sequence(i);

            auto begin = std::chrono::high_resolution_clock::now();
            auto result = index->search(q_data, q_len, k, ef);
            auto end = std::chrono::high_resolution_clock::now();
            time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

            assert(result.size() <= k);
            while (result.size() > 0) {
                int id = result.top().second;
                result.pop();
                if (groundtruth[i].find(id) != groundtruth[i].end()) {
                    hit++;
                }
            }

            total += groundtruth[i].size();
            q_num++;
        }

        return {ef, time, hit, total, q_num};
    }

    void save_records(std::vector<QueryRecord>& records) {
        std::string csv_name = index_name + "_search.csv";
        fs::path csv_path = fs::path("../log") / data_dir / csv_name;
        fs::create_directories(csv_path.parent_path());

        std::ofstream ofs(csv_path);
        cerr_if(!ofs.is_open(), "Failed to open " + csv_name);

        ofs << "ef,time,hit,total,q_num" << std::endl;
        for (const auto& r : records) {
            ofs << r.ef << "," << r.time << "," << r.hit << "," << r.total << "," << r.q_num << std::endl;
        }

        ofs.close();
        std::cout << "Query records written to " << csv_path << std::endl;
    }
};

} // namespace vss