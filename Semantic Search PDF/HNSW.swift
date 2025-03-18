//
//  HNSW.swift
//  Semantic Search PDF
//
//  Created by Aditya Jadhav
//

import Foundation

struct PriorityQueue<T> {
    private var elements: [(T, Double)]
    private let sort: (Double, Double) -> Bool

    init(sort: @escaping (Double, Double) -> Bool) {
        self.elements = []
        self.sort = sort
    }

    mutating func push(_ element: T, _ priority: Double) {
        elements.append((element, priority))
        elements.sort { sort($0.1, $1.1) }  // Sort based on provided order
    }

    mutating func pop() -> T? {
        return elements.isEmpty ? nil : elements.removeFirst().0
    }

    func peek() -> T? {
        return elements.first?.0
    }

    var isEmpty: Bool {
        return elements.isEmpty
    }
}


class HNSW {
    var nodes: [Int: HNSWNode] = [:]
    var entryPoint: HNSWNode?
    let M: Int
    let Mmax: Int
    let efConstruction: Int
    let mL: Double

    init(M: Int, Mmax: Int, efConstruction: Int, mL: Double) {
        self.M = M
        self.Mmax = Mmax
        self.efConstruction = efConstruction
        self.mL = mL
    }
}


class HNSWNode {
    let id: Int
    let level: Int
    let vector: [Double]
    var neighbors: [Int: [HNSWNode]] // Dictionary for layer-wise neighbors

    init(id: Int, level: Int, vector: [Double]) {
        self.id = id
        self.level = level
        self.vector = vector
        self.neighbors = [:]
        for i in 0...level {
            self.neighbors[i] = []
        }
    }
}


func cosineDistance(_ vectorA: [Double], _ vectorB: [Double]) -> Double {
        let dotProduct = zip(vectorA, vectorB).map(*).reduce(0, +)
        let magnitudeA = sqrt(vectorA.map { $0 * $0 }.reduce(0, +))
        let magnitudeB = sqrt(vectorB.map { $0 * $0 }.reduce(0, +))
        return dotProduct / (magnitudeA * magnitudeB)
    }




func search_layer(q: [Double], ep: HNSWNode, ef: Int, lc: Int) -> [HNSWNode] {
    var v = Set<Int>()
    var C = PriorityQueue<HNSWNode>(sort: { $0.1 < $1.1 }) // Min-Heap
    var result = PriorityQueue<HNSWNode>(sort: { $0.1 > $1.1 }) // Max-Heap
    
    C.push(ep, cosineDistance(ep.vector, q))
    v.insert(ep.id)
    result.push(ep, cosineDistance(ep.vector, q))
    
    while C.isEmpty() == false {
        let (curDist, cur) = C.pop()!
        
        if (result.isEmpty() == false) {
            let worst = result.peek()!
            
            if (cosineDistance(q, cur.vector) > cosineDistance(q, worst.vector)) {
                break
            }
        }
        
        for neighbor in cur.neighbors[lc] ?? [] {
            if (v.contains(neighbor.id) == false) {
                v.insert(neighbor.id)
                let dist = cosineDistance(q, neighbor.vector)
                
                let f = result.peek()
                let worstDist = f != nil ? cosineDistance(q, f!.vector) : Double.infinity
                
                if (dist < worstDist || result.elements.count < ef) {
                    C.push(neighbor, dist)
                    result.push(neighbor, dist)
                    
                    if (result.elements.count > ef) {
                        _ = result.pop()
                    }
                }
            }
        }
    }
    
    return result.elements.map { $0.0 }
}

func insertHNSW(hnsw: inout HNSW, q: [Double], M: Int, Mmax: Int, efConstruction: Int, mL: Double) {
    let l = Int(-log(Double.random(in: 0...1)) * mL) // Assign level based on probability
    let newNode = HNSWNode(id: hnsw.nodes.count, level: l, vector: q) // Create new node

    if hnsw.nodes.isEmpty {
        hnsw.entryPoint = newNode
        hnsw.nodes[newNode.id] = newNode
        return
    }

    var ep = hnsw.entryPoint!
    let L = ep.level

    // **Step 1: Find entry point at the highest layer**
    for lc in stride(from: L, through: l + 1, by: -1) {
        let W = search_layer(q: q, ep: ep, ef: 1, lc: lc)
        if let nearest = W.first {
            ep = nearest
        }
    }

    // **Step 2: Insert into layers down to layer 0**
    for lc in stride(from: min(L, l), through: 0, by: -1) {
        let W = search_layer(q: q, ep: ep, ef: efConstruction, lc: lc)
        let neighbors = select_neighbors(q: newNode, candidates: W, M: M, lc: lc) // Uses Alg 3 or 4

        newNode.neighbors[lc] = Array(neighbors)

        // **Bidirectional connections**
        for neighbor in neighbors {
            neighbor.neighbors[lc]?.append(newNode)
            if neighbor.neighbors[lc]!.count > Mmax {
                let reduced_neighbors = select_neighbors(q: neighbor, candidates: neighbor.neighbors[lc]!, M: Mmax, lc: lc)
                neighbor.neighbors[lc] = reduced_neighbors
            }
        }

        ep = W.first ?? ep
    }

    if l > L {
        hnsw.entryPoint = newNode
    }

    hnsw.nodes[newNode.id] = newNode
}


func knnSearch(query: [Double], k: Int, ef: Int) -> [HNSWNode] {
    guard let ep = entryPoint else { return [] }
    
    let L = ep.level
    var epCurrent = ep

    for lc in stride(from: L, through: 1, by: -1) {
        let W = searchLayer(query: query, entryPoint: epCurrent, ef: 1, layer: lc)
        if let nearest = W.first {
            epCurrent = nearest
        }
    }

    let finalCandidates = searchLayer(query: query, entryPoint: epCurrent, ef: ef, layer: 0)
    return Array(finalCandidates.prefix(k))
}

  
func findTopNSimilarEmbeddings(queryEmbedding: [Double], topN: Int) -> [EmbeddingWithPosition] {
    let nearestNodes = hnsw.knnSearch(query: queryEmbedding, k: topN, ef: 50)

    return nearestNodes.compactMap { node in
        pdfTextWithPosition.first { $0.pageNumber == node.id }.map {
            EmbeddingWithPosition(embedding: node.vector, position: $0)
        }
    }
}

func fit(embeddings: [EmbeddingWithPosition]) {
    for embedding in embeddings {
        insertHNSW(hnsw: &hnsw, q: embedding.embedding, M: hnsw.M, Mmax: hnsw.Mmax, efConstruction: hnsw.efConstruction, mL: hnsw.mL)
    }
}

func createHNSW(M: Int = 5, Mmax: Int = 10, efConstruction: Int = 200, mL: Double = 1.0) {
    hnsw = HNSW(M: M, Mmax: Mmax, efConstruction: efConstruction, mL: mL)
}
