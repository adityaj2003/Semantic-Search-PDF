#include "hnsw_wrapper.h"
#include "hnsw_mmap.hpp"

void* create_hnsw(int maxLevel, int ef, double mL) {
    return new HNSW(maxLevel, ef, mL);
}

void insert_hnsw(void* hnsw, const float* vector, int dimension, int id) {
    std::vector<float> vec(vector, vector + dimension);
    static_cast<HNSW*>(hnsw)->insert(vec, id);
}

void knn_search_hnsw(void* hnsw, const float* query, int dimension, int k, int* resultIds) {
    std::vector<float> queryVec(query, query + dimension);
    std::vector<HNSWNode*> results = static_cast<HNSW*>(hnsw)->knnSearch(queryVec, k);

    for (size_t i = 0; i < results.size(); ++i) {
        resultIds[i] = results[i]->getId();
    }
}

void free_hnsw(void* hnsw) {
    delete static_cast<HNSW*>(hnsw);
}

// Setup mmap
void setup_mmap() {
    ::setup_mmap();
}

// Write HNSW index to mmap
void write_hnsw_to_mmap(void* hnsw) {
    ::write_index_to_mmap(*static_cast<HNSW*>(hnsw));
}

// Read HNSW index from mmap
void read_hnsw_from_mmap(void* hnsw) {
    ::read_index_from_mmap(*static_cast<HNSW*>(hnsw));
}

