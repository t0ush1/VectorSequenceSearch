#pragma once

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>

#include "dataset.h"
#include "index.h"

namespace vss {

class IVFPQIndex : public RerankIndex {
public:
    int nlist;  // 倒排表数量
    int m;      // PQ分块数
    int nbits;  // 每个子量化器bit数
    int nprobe; // 搜索时访问的倒排表数量

    faiss::IndexIVFPQ* index;
    faiss::IndexFlatL2* quantizer;

    IVFPQIndex(int dim, int nlist = 100, int m = 8, int nbits = 8, int nprobe = 10)
        : RerankIndex(dim), nlist(nlist), m(m), nbits(nbits), nprobe(nprobe) {}

    ~IVFPQIndex() {
        delete index;
        delete quantizer;
    }

    void build_index() override {
        quantizer = new faiss::IndexFlatL2(dim);
        index = new faiss::IndexIVFPQ(quantizer, dim, nlist, m, nbits);
        index->nprobe = nprobe;

        index->train(base_vec_num, base_data[0]);
        index->add(base_vec_num, base_data[0]);
    }

    void search_index(std::unordered_set<int>& candidates, const float* q_data, int q_len, int k) override {
        std::vector<float> D(k * q_len);
        std::vector<faiss::idx_t> I(k * q_len);
        index->search(q_len, q_data, k, D.data(), I.data());
        for (auto idx : I) {
            candidates.insert(label_to_base[idx]);
        }
    }
};

} // namespace vss
