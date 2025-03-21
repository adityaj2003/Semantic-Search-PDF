#include "hnsw_mmap.hpp"




std::mt19937 rng(1000);
std::uniform_real_distribution<double> uniform(0.0, 1.0);

char *mmap_obj;
int fd;

void handle(const char *msg) {
    perror(msg);
    std::cerr << "Failed in: " << msg << std::endl;
    exit(EXIT_FAILURE);
}


void setup_mmap() {
    std::cerr << "Start" << std::endl;
    fd = open(MMAP_FILE, O_RDWR | O_CREAT, 0666);
    if (fd == -1) handle("open");
    struct stat sb;
    fstat(fd, &sb);
    std::cerr << "First " << std::endl;
    if (sb.st_size == 0) {
        if (ftruncate(fd, MAX_MMAP_LENGTH) == -1) handle("ftruncate");
    }
    std::cerr << "Second " <<  std::endl;
    mmap_obj = (char *)mmap(NULL, MAX_MMAP_LENGTH, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mmap_obj == MAP_FAILED) handle("mmap");
}

void write_index_to_mmap(HNSW &hnsw) {
    std::vector<HNSWNode*> nodes = hnsw.getAllNodes();
    char *ptr = mmap_obj;  // Start writing from mmap memory
    size_t numNodes = nodes.size();
    memcpy(ptr, &numNodes, sizeof(size_t));
    ptr += sizeof(size_t);
    for (HNSWNode *node : nodes) {
        int id = node->getId();
        int level = node->getLevel();
        std::vector<float> embedding = node->getVector();
        memcpy(ptr, &id, sizeof(int));
        ptr += sizeof(int);
        memcpy(ptr, &level, sizeof(int));
        ptr += sizeof(int);
        size_t vecSize = embedding.size();
        memcpy(ptr, &vecSize, sizeof(size_t));
        ptr += sizeof(size_t);
        memcpy(ptr, embedding.data(), vecSize * sizeof(float));
        ptr += vecSize * sizeof(float);



        for (int l = 0; l <= level; l++) {
            std::vector<HNSWNode*> neighbors = node->getNeighbors(l);
            size_t numNeighbors = neighbors.size();
            memcpy(ptr, &numNeighbors, sizeof(size_t));
            ptr += sizeof(size_t);
            for (HNSWNode *neighbor : neighbors) {
                int neighborId = neighbor->getId();
                memcpy(ptr, &neighborId, sizeof(int));
                ptr += sizeof(int);
            }
        }
    }
}



void read_index_from_mmap(HNSW &hnsw) {
    char *ptr = mmap_obj;
    size_t numNodes;
    memcpy(&numNodes, ptr, sizeof(size_t));
    ptr += sizeof(size_t);
    std::unordered_map<int, HNSWNode*> nodeMap;

    for (size_t i = 0; i < numNodes; i++) {
        int id, level;
        memcpy(&id, ptr, sizeof(int));
        ptr += sizeof(int);
        memcpy(&level, ptr, sizeof(int));
        ptr += sizeof(int);
        size_t vecSize;
        memcpy(&vecSize, ptr, sizeof(size_t));
        ptr += sizeof(size_t);
        std::vector<float> embedding(vecSize);
        memcpy(embedding.data(), ptr, vecSize * sizeof(float));
        ptr += vecSize * sizeof(float);


        HNSWNode *node = new HNSWNode(id, level);
        node->setVector(embedding);
        nodeMap[id] = node;
        hnsw.insert(node->getVector(), id);
    }

    for (size_t i = 0; i < numNodes; i++) {
        HNSWNode *node = nodeMap[i];

        for (int l = 0; l <= node->getLevel(); l++) {
            size_t numNeighbors;
            memcpy(&numNeighbors, ptr, sizeof(size_t));
            ptr += sizeof(size_t);

            for (size_t j = 0; j < numNeighbors; j++) {
                int neighborId;
                memcpy(&neighborId, ptr, sizeof(int));
                ptr += sizeof(int);

                if (nodeMap.find(neighborId) != nodeMap.end()) {
                    node->addNeighbor(l, nodeMap[neighborId]);
                }
            }
        }
    }
}



HNSWNode::HNSWNode(int id, int level) : id(id), level(level) {}

int HNSWNode::getId() const { return id; }

int HNSWNode::getLevel() const { return level; }

std::vector<float> HNSWNode::getVector() const { return embedding; }

void HNSWNode::setVector(const std::vector<float>& vec) { embedding = vec; }

std::vector<HNSWNode*> HNSWNode::getNeighbors(int level) const {
    if (neighbors.find(level) != neighbors.end()) {
        return neighbors.at(level);
    }
    return {};
}

void HNSWNode::addNeighbor(int level, HNSWNode* neighbor) {
    neighbors[level].push_back(neighbor);
}

int HNSWNode::getNeighborsSize(int level) const {
    auto it = neighbors.find(level);
    return (it == neighbors.end()) ? 0 : it->second.size();
}

void HNSWNode::clearNeighbors(int level) {
    if (neighbors.find(level) != neighbors.end()) neighbors[level].clear();
}

HNSW::HNSW(int maxLevel, int ef, double mL) : maxLevel(maxLevel), ef(ef), mL(mL) {
    entryPoint = nullptr;
    maxM = {32, 16, 16, 12, 8};
}

int HNSW::randomLevel() {
    double r = uniform(rng);
    return (int) std::floor(-std::log(r) * mL);
}

std::vector<HNSWNode*> HNSW::getAllNodes() {
    std::vector<HNSWNode*> result;
    for (auto& kv : nodes) {
        result.push_back(kv.second);
    }
    return result;
}

float HNSW::distance(const std::vector<float>& a, const std::vector<float>& b) {
    float dist = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(dist);
}

std::vector<HNSWNode*> HNSW::select_neighbors(const std::vector<float>& query, int k, const std::vector<HNSWNode*>& candidates) {
    std::vector<HNSWNode*> result;
    std::priority_queue<
        std::pair<float, HNSWNode*>,
        std::vector<std::pair<float, HNSWNode*>>,
        std::greater<std::pair<float, HNSWNode*>>
    > pq;

    for (auto candidate : candidates) {
        float dist = distance(query, candidate->getVector());
        pq.push(std::make_pair(dist, candidate));
        if (pq.size() > k) {
            pq.pop();
        }
    }

    while (!pq.empty()) {
        result.push_back(pq.top().second);
        pq.pop();
    }

    return result;
}

std::vector<HNSWNode*> HNSW::search_layer(const std::vector<float>& query, std::vector<HNSWNode*> ep, int ef, int level) {
    std::unordered_set<HNSWNode*> visited;
    std::priority_queue<std::pair<float, HNSWNode*>> result;
    std::priority_queue<
        std::pair<float, HNSWNode*>,
        std::vector<std::pair<float, HNSWNode*>>,
        std::greater<std::pair<float, HNSWNode*>>
    > candidates;

    for (auto node : ep) {
        float nodeDist = distance(query, node->getVector());
        visited.insert(node);
        candidates.push({nodeDist, node});
        result.push({nodeDist, node});
    }

    while (!candidates.empty()) {
        auto current = candidates.top().second;
        auto dist = candidates.top().first;
        candidates.pop();

        if (!result.empty() && result.top().first < dist) {
            break;
        }

        for (auto neighbor : current->getNeighbors(level)) {
            if (visited.find(neighbor) != visited.end()) continue;
            visited.insert(neighbor);

            float curDist = distance(query, neighbor->getVector());
            if (result.size() >= (size_t)ef && curDist >= result.top().first) continue;

            candidates.push({curDist, neighbor});
            result.push({curDist, neighbor});

            if (result.size() > ef) result.pop();
        }
    }

    std::vector<HNSWNode*> ret;
    while (!result.empty()) {
        ret.push_back(result.top().second);
        result.pop();
    }

    std::reverse(ret.begin(), ret.end());
    return ret;
}

std::vector<HNSWNode*> HNSW::knnSearch(const std::vector<float>& query, int k) {
    std::vector<HNSWNode*> ep;
    if (entryPoint != nullptr) ep.push_back(entryPoint);

    for (int lc = maxLevel; lc > 0; lc--) {
        auto w = search_layer(query, ep, 1, lc);
        if (!w.empty()) ep = {w[0]};
    }

    return search_layer(query, ep, k, 0);
}


void HNSW::insert(const std::vector<float>& query, int id) {
    int level = randomLevel();
    HNSWNode* newNode = new HNSWNode(id, level);
    newNode->setVector(query);
    nodes[id] = newNode;

    if (!entryPoint) entryPoint = newNode;

    std::vector<HNSWNode*> ep = {entryPoint};
    for (int lc = maxLevel; lc > level; lc--) {
        auto w = search_layer(query, ep, 1, lc);
        if (!w.empty()) ep = {w[0]};
    }

    for (int lc = std::min(maxLevel, level); lc >= 0; lc--) {
        auto w = search_layer(query, ep, ef, lc);
        auto neighbors = select_neighbors(query, maxM[lc], w);

        for (auto neighbor : neighbors) {
            newNode->addNeighbor(lc, neighbor);
            neighbor->addNeighbor(lc, newNode);
        }

        ep = w;
    }

    if (level > maxLevel) entryPoint = newNode;
}
