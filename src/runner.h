#pragma once
#include <chrono>
#include <filesystem>
#include <functional>
#include <unordered_map>
#include <unordered_set>

#include "baselines/brute_force.h"
#include "baselines/hnsw.h"
#include "baselines/ivfpq.h"
#include "dataset.h"
#include "index.h"
#include "sequence_graph.h"
#include "set_graph.h"

#include "baselines/my_hnsw_rerank.h"

namespace vss {
class VSSRunner {
public:
    struct QueryRecord {
        int ef;
        int q_num;

        size_t time;

        int hit;
        int total;

        long hops;
        long dist_comps;
    };

    std::string log_time;

    int dim;
    std::string sim_metric;
    std::string data_dir;
    std::string index_name;

    VSSDataset* base_dataset;
    VSSDataset* query_dataset;
    std::vector<std::unordered_set<int>> groundtruth;

    VSSIndex* index;
    std::vector<int> efs;

    VSSRunner(int dim, std::string sim_metric, std::string data_dir, std::string index_name)
        : dim(dim), sim_metric(sim_metric), data_dir(data_dir), index_name(index_name) {
        fs::path data_path = fs::path("../datasets") / data_dir;
        base_dataset = new VSSDataset(dim, data_path / "base.fvecs", data_path / "base.lens");
        query_dataset = new VSSDataset(dim, data_path / "query.fvecs", data_path / "query.lens");
        groundtruth = read_groundtruth(data_path / "groundtruth-cpp.ivecs");

        if (index_name == "brute_force") {
            index = new BruteForceIndex(dim, sim_metric);
            efs = {0};
        } else if (index_name == "hnsw") {
            index = new HNSWIndex(dim, sim_metric, 16, 200);
            efs = {10, 20, 50, 100, 200, 500, 1000};
        } else if (index_name == "ivfpq") {
            index = new IVFPQIndex(dim, sim_metric, 100, 8, 8);
            efs = {10, 20, 50, 100, 200, 500, 1000};
        } else if (index_name == "set") {
            index = new SetGraphIndex<float>(dim, sim_metric, 16, 200);
            efs = {10, 20, 50, 100, 200, 500, 1000, 2000};
        } else if (index_name == "sequence") {
            index = new SequenceGraphIndex<float, int>(dim, sim_metric, 16, 200);
            efs = {10, 20, 50, 100, 200, 500, 1000, 2000};
        } else if (index_name == "my_hnsw") {
            index = new MyHNSWIndex(dim, sim_metric, 16, 200);
            efs = {10, 20, 50, 100, 200, 500, 1000};
        } else {
            std::cerr << "Unknown index: " << index_name << std::endl;
            std::exit(-1);
        }

        std::time_t t = std::time(nullptr);
        char buf[16];
        std::strftime(buf, sizeof(buf), "%y%m%d-%H%M%S", std::localtime(&t));
        log_time = buf;
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
        std::vector<QueryRecord> records(efs.size());
        int k = groundtruth[0].size();

        for (int i = 0; i < efs.size(); i++) {
            run_search_once(k, efs[i], records[i]);
        }

        for (const auto& r : records) {
            std::cout << "EF: " << r.ef << std::endl;
            std::cout << "Time: " << r.time << " us, Avg Time: " << r.time / r.q_num << " us" << std::endl;
            std::cout << "Recall: " << r.hit << "/" << r.total << "=" << r.hit * 1.0 / r.total << std::endl;
            std::cout << "Hops: " << r.hops << ", Avg Hops: " << r.hops / r.q_num << std::endl;
            std::cout << "Dist Comps: " << r.dist_comps << ", Avg Dist Comps: " << r.dist_comps / r.q_num << std::endl;
            std::cout << std::endl;
        }

        save_records(records);
    }

    void run_search_once(int k, int ef, QueryRecord& record) {
        record.ef = ef;
        index->reset_metric();

        for (int i = 0; i < query_dataset->seq_num; i++) {
            auto [q_data, q_len] = query_dataset->get_sequence(i);

            auto begin = std::chrono::high_resolution_clock::now();
            auto result = index->search(q_data, q_len, k, ef);
            auto end = std::chrono::high_resolution_clock::now();
            record.time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

            record.hops += index->get_metric("hops");
            record.dist_comps += index->get_metric("dist_comps");

            assert(result.size() <= k);
            while (result.size() > 0) {
                int id = result.top().second;
                result.pop();
                if (groundtruth[i].find(id) != groundtruth[i].end()) {
                    record.hit++;
                }
            }

            record.total += groundtruth[i].size();
            record.q_num++;
        }
    }

    void save_records(std::vector<QueryRecord>& records) {
        std::string csv_name = index_name + "-search-" + log_time + ".csv";
        fs::path csv_path = fs::path("../log") / data_dir / csv_name;
        fs::create_directories(csv_path.parent_path());

        std::ofstream ofs(csv_path);
        cerr_if(!ofs.is_open(), "Failed to open " + csv_name);

        ofs << "ef,time,hit,total,q_num,hops,distance_computations" << std::endl;
        for (const auto& r : records) {
            ofs << r.ef << "," << r.time << "," << r.hit << "," << r.total << "," << r.q_num << "," << r.hops << ","
                << r.dist_comps << std::endl;
        }

        ofs.close();
        std::cout << "Query records written to " << csv_path << std::endl;
    }
};

} // namespace vss