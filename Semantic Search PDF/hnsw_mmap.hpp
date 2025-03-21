#ifndef HNSW_MMAP_HPP
#define HNSW_MMAP_HPP
#include <iostream>
#include <cmath>
#include <queue>
#include <unordered_set>
#include <random>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unordered_map>
#define MMAP_FILE "hnsw_index.bin"
#define MAX_MMAP_LENGTH 1024 * 1024 * 250  // 250MB

extern char *mmap_obj;
extern int fd;

void setup_mmap();
void write_index_to_mmap(class HNSW &hnsw);
void read_index_from_mmap(class HNSW &hnsw);

class HNSWNode {
public:
    HNSWNode(int id, int level);
    int getId() const;
    int getLevel() const;
    std::vector<float> getVector() const;
    void setVector(const std::vector<float>& vec);
    std::vector<HNSWNode*> getNeighbors(int level) const;
    void addNeighbor(int level, HNSWNode* neighbor);
    int getNeighborsSize(int level) const;
    void clearNeighbors(int level);

private:
    int id;
    int level;
    std::unordered_map<int, std::vector<HNSWNode*>> neighbors;
    std::vector<float> embedding;
};

class HNSW {
public:
    HNSW(int maxLevel, int ef, double mL);
    std::vector<HNSWNode*> getAllNodes();
    std::vector<HNSWNode*> knnSearch(const std::vector<float>& query, int k);
    void insert(const std::vector<float>& query, int id);

private:
    std::unordered_map<int, HNSWNode*> nodes;
    HNSWNode* entryPoint;
    const int maxLevel;
    std::vector<int> maxM;
    int ef;
    double mL;
    int randomLevel();
    float distance(const std::vector<float>& a, const std::vector<float>& b);
    std::vector<HNSWNode*> search_layer(const std::vector<float>& query, std::vector<HNSWNode*> ep, int ef, int level);
    std::vector<HNSWNode*> select_neighbors(const std::vector<float>& query, int k, const std::vector<HNSWNode*>& candidates);
};

#endif // HNSW_MMAP_HPP


