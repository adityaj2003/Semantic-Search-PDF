import Accelerate

import Accelerate

class Quantizer {
    var numSubspaces: Int
    var numClusters: Int
    var maxIterations: Int
    var centroids: [[[Float]]]

    init(numSubspaces: Int, numClusters: Int, maxIterations: Int) {
        self.numSubspaces = numSubspaces
        self.numClusters = numClusters
        self.maxIterations = maxIterations
        self.centroids = []
    }

    func fit(embeddings: [[Float]]) {
        self.centroids = (0..<numSubspaces).map { subspaceIndex in
            let subspaceEmbeddings = embeddings.map { Array($0[subspaceIndex * (384 / numSubspaces)..<((subspaceIndex + 1) * (384 / numSubspaces))]) }
            return kMeansClustering(data: subspaceEmbeddings, k: numClusters, maxIterations: maxIterations)
        }
    }

    func quantize(embedding: [Float]) -> [Int] {
        let subvectors = splitEmbedding(embedding, numSubspaces: numSubspaces)
        return zip(subvectors, centroids).map { quantizeRec(subvector: $0.0, centroids: $0.1) }
    }
    func splitEmbedding(_ embedding: [Float], numSubspaces: Int) -> [[Float]] {
        let subvectorSize = embedding.count / self.numSubspaces
        var subvectors: [[Float]] = []
        
        for i in 0..<numSubspaces {
            let start = i * subvectorSize
            let end = start + subvectorSize
            let subvector = Array(embedding[start..<end])
            subvectors.append(subvector)
        }
        
        return subvectors
    }

    func kMeansClustering(data: [[Float]], k: Int, maxIterations: Int) -> [[Float]] {
        let n = data.count
        let d = data[0].count
        var centroids: [[Float]] = (0..<k).map { _ in (0..<d).map { _ in Float.random(in: -1...1) } }
        
        for _ in 0..<maxIterations {
            var clusters: [[[Float]]] = Array(repeating: [], count: k)
            
            for point in data {
                let distances = centroids.map { centroid -> Float in
                    let diff = zip(point, centroid).map { $0 - $1 }
                    return sqrt(diff.reduce(0, { $0 + $1 * $1 }))
                }
                let nearestCentroidIndex = distances.enumerated().min(by: { $0.element < $1.element })!.offset
                clusters[nearestCentroidIndex].append(point)
            }
            
            for i in 0..<k {
                if clusters[i].isEmpty { continue }
                let sum = clusters[i].reduce(Array(repeating: 0.0, count: d)) { zip($0, $1).map(+) }
                centroids[i] = sum.map { $0 / Float(clusters[i].count) }
            }
        }
        
        return centroids
    }

    func quantizeRec(subvector: [Float], centroids: [[Float]]) -> Int {
        let distances = centroids.map { centroid -> Float in
            let diff = zip(subvector, centroid).map { $0 - $1 }
            return sqrt(diff.reduce(0, { $0 + $1 * $1 }))
        }
        return distances.enumerated().min(by: { $0.element < $1.element })!.offset
    }

    func quantizeEmbedding(embedding: [Float], centroids: [[[Float]]]) -> [Int] {
        let subvectors = splitEmbedding(embedding, numSubspaces: centroids.count)
        return zip(subvectors, centroids).map { quantizeRec(subvector: $0.0, centroids: $0.1) }
    }

}

