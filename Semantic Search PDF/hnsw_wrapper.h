#ifndef HNSW_WRAPPER_H
#define HNSW_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif


void* create_hnsw(int maxLevel, int ef, double mL);
void insert_hnsw(void* hnsw, const float* vector, int dimension, int id);
void knn_search_hnsw(void* hnsw, const float* query, int dimension, int k, int* resultIds);
void free_hnsw(void* hnsw);
void setup_mmap();
void write_hnsw_to_mmap(void* hnsw);
void read_hnsw_from_mmap(void* hnsw);

#ifdef __cplusplus
}
#endif

#endif // HNSW_WRAPPER_H
