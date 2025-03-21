import SwiftUI
import PDFKit
import CoreML
import AppKit
import hnsw_mmap

@available(macOS 14.0, *)
class ViewModel: ObservableObject {
    @Published var selectedFileURL: URL?
    @Published var isFileImporterPresented: Bool = true
    @Published var embeddingsWithPositions: [EmbeddingWithPosition] = []
    @Published var pdfTextWithPosition: [TextWithPosition] = []
    @Published var pdfKitView: PDFKitView?
    @Published var topMatches: [EmbeddingWithPosition] = []
    @Published var hnsw = create_hnsw(5,50, 0.3)
    @Published var isSearching = false
    @Published var isHidden = true
    @Published var isHiddenText = false
    @Published var searchText = ""
    @Published var indexedPages = 0
    var id = 0

    func handleFileImport(result: Result<URL, Error>) {
        switch result {
        case .success(let url):
            DispatchQueue.main.async {
                self.selectedFileURL = url
            }
        case .failure(let error):
            print("Error selecting file: \(error.localizedDescription)")
        }
    }


    func handleFileChange(newURL: URL?) {
        if let pdfURL = newURL {
            DispatchQueue.global(qos: .userInitiated).async {
                DispatchQueue.main.async {
                    self.indexedPages = 0
                    self.pdfKitView = PDFKitView(url: pdfURL)
                    self.isHiddenText = false
                    self.embeddingsWithPositions = []
                    self.pdfTextWithPosition = []
                    self.topMatches = []
                    self.hnsw = create_hnsw(5,50,0.3)
                }
                let pdfTextWithPosition = self.extractTextWithPosition(from: pdfURL)
                self.pdfTextWithPosition = pdfTextWithPosition
            }
            
        }
    }

    func tokenizeAndEmbed(textChunks: [TextWithPosition], tokenizer: BertTokenizer, model: MiniLM_V6) -> [EmbeddingWithPosition] {
        var embeddingsWithPositions: [EmbeddingWithPosition] = []

        for chunk in textChunks {
            DispatchQueue.main.async {
                self.indexedPages = chunk.pageNumber
            }
            let tokenIds = tokenizer.tokenizeToIds(text: chunk.text)
            let attentionMask = Array(repeating: 1, count: tokenIds.count)
            let tokenTypeIds = Array(repeating: 0, count: tokenIds.count)

            guard let inputIdsArray = try? MLMultiArray(shape: [1, NSNumber(value: tokenIds.count)], dataType: .int32),
                  let attentionMaskArray = try? MLMultiArray(shape: [1, NSNumber(value: tokenIds.count)], dataType: .int32),
                  let tokenTypeIdsArray = try? MLMultiArray(shape: [1, NSNumber(value: tokenIds.count)], dataType: .int32) else {
                print("Failed to create MLMultiArray")
                continue
            }

            for (index, tokenId) in tokenIds.enumerated() {
                inputIdsArray[index] = NSNumber(value: tokenId)
                attentionMaskArray[index] = NSNumber(value: attentionMask[index])
                tokenTypeIdsArray[index] = NSNumber(value: tokenTypeIds[index])
            }

            let input = MiniLM_V6Input(input_ids: inputIdsArray, attention_mask: attentionMaskArray, token_type_ids: tokenTypeIdsArray)

            do {
                let prediction = try model.prediction(input: input)
                let modelOutput = (0..<prediction.Identity.count).map { prediction.Identity[$0].doubleValue }

                // Reshape the model output
                let embeddingDim = 384 // Assuming MiniLM-L6-v2 with 384 dimensions
                let sequenceLength = modelOutput.count / embeddingDim
                let reshapedOutput = stride(from: 0, to: modelOutput.count, by: embeddingDim).map {
                    Array(modelOutput[$0..<min($0 + embeddingDim, modelOutput.count)])
                }

                // Perform mean pooling
                let pooledEmbedding = meanPooling(reshapedOutput, attentionMask: attentionMask.map { Double($0) })

                // Normalize the pooled embedding
                let normalizedEmbedding = normalizeEmbedding(pooledEmbedding)

                let embeddingWithPosition = EmbeddingWithPosition(embedding: normalizedEmbedding, position: chunk, id : id)
                id += 1
                
                embeddingsWithPositions.append(embeddingWithPosition)

            } catch {
                print("Failed to get Embeddings: \(error)")
            }
        }

        return embeddingsWithPositions
    }

    func meanPooling(_ modelOutput: [[Double]], attentionMask: [Double]) -> [Double] {
        let embeddingDim = modelOutput[0].count
        var pooledEmbedding = [Double](repeating: 0.0, count: embeddingDim)
        var validTokenCount = 0.0
        
        for (tokenEmbedding, mask) in zip(modelOutput, attentionMask) {
            for (value, dim) in zip(tokenEmbedding, 0..<embeddingDim) {
                pooledEmbedding[dim] += value * mask
            }
            validTokenCount += mask
        }
        
        return pooledEmbedding.map { $0 / max(validTokenCount, 1e-9) }
    }

    func normalizeEmbedding(_ embedding: [Double]) -> [Double] {
        let norm = sqrt(embedding.reduce(0) { $0 + $1 * $1 })
        return embedding.map { $0 / norm }
    }

    func splitTextIntoSentences(text: String) -> [String] {
                let pattern = "(?<!\\b[A-Z]\\.|Mr\\.|Mrs\\.|Ms\\.|Dr\\.|\\d\\.\\d)\\s*[\\.\\!\\?]\\s+"
                var sentences: [String] = []
                do {
                    let regex = try NSRegularExpression(pattern: pattern, options: [])
                    let matches = regex.matches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count))
                    
                    var lastRangeEnd = text.startIndex
                    for match in matches {
                        let range = Range(match.range(at: 0), in: text)!
                        let sentence = text[lastRangeEnd..<range.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
                        if !sentence.isEmpty {
                            sentences.append(String(sentence))
                        }
                        lastRangeEnd = range.upperBound
                    }
                    
                    // Add the last sentence
                    let lastSentence = text[lastRangeEnd..<text.endIndex].trimmingCharacters(in: .whitespacesAndNewlines)
                    if !lastSentence.isEmpty {
                        sentences.append(String(lastSentence))
                    }
                } catch {
                    print("Invalid regex pattern")
                }
                
                return sentences
            }

    func extractTextWithPosition(from url: URL) -> [TextWithPosition] {
        guard let pdfDocument = PDFDocument(url: url) else { return [] }
        var textWithPositions: [TextWithPosition] = []

        for pageIndex in 0..<pdfDocument.pageCount {
            guard let page = pdfDocument.page(at: pageIndex) else { continue }

            if let pageText = page.string {
                let sentences = splitTextIntoSentences(text: pageText)
                for sentence in sentences {
                    if let selection = page.selection(for: NSRange(location: pageText.distance(from: pageText.startIndex, to: pageText.range(of: sentence)!.lowerBound), length: sentence.count)), selection.string!.count > 15 {
                        let bounds = selection.bounds(for: page)
                        let position = TextWithPosition(
                            text: sentence,
                            pageNumber: pageIndex,
                            bounds: bounds
                        )
                        textWithPositions.append(position)
                    }
                }
            }
        }
        
        DispatchQueue.main.async {
            self.isHiddenText = false
        }
        
        let tokenizer = BertTokenizer()
        guard let model = try? MiniLM_V6(configuration: .init()) else {
            print("Failed to load model")
            return textWithPositions
        }

        let embeddingsWithPositions = tokenizeAndEmbed(textChunks: textWithPositions, tokenizer: tokenizer, model: model)
        fit(embeddings: embeddingsWithPositions)

        DispatchQueue.main.async {
            self.embeddingsWithPositions = embeddingsWithPositions
            self.isHiddenText = true
    
        }

        return textWithPositions
    }


    

    func highlightSelectedMatch(match: EmbeddingWithPosition, pdfView: PDFView?) {
            guard let pdfView = pdfView,
                  let document = pdfView.document,
                  let page = document.page(at: match.position.pageNumber) else {
                print("Failed to highlight match: PDFView, document, or page is nil")
                return
            }
            pdfView.go(to: match.position.bounds, on: page)
        }
    
    func highlightTopMatches(pdfView: PDFView?) {
        let tokenizer = BertTokenizer()
        guard let model = try? MiniLM_V6(configuration: .init()) else {
            print("Failed to load model")
            return
        }

        let tokenIds = tokenizer.tokenizeToIds(text: searchText)
        let attentionMask = Array(repeating: 1, count: tokenIds.count)
        let tokenTypeIds = Array(repeating: 0, count: tokenIds.count)

        guard let inputIdsArray = try? MLMultiArray(shape: [1, NSNumber(value: tokenIds.count)], dataType: .int32),
              let attentionMaskArray = try? MLMultiArray(shape: [1, NSNumber(value: tokenIds.count)], dataType: .int32),
              let tokenTypeIdsArray = try? MLMultiArray(shape: [1, NSNumber(value: tokenIds.count)], dataType: .int32) else {
            print("Failed to create MLMultiArray")
            return
        }

        for (index, tokenId) in tokenIds.enumerated() {
            inputIdsArray[index] = NSNumber(value: tokenId)
            attentionMaskArray[index] = NSNumber(value: attentionMask[index])
            tokenTypeIdsArray[index] = NSNumber(value: tokenTypeIds[index])
        }

        let input = MiniLM_V6Input(input_ids: inputIdsArray, attention_mask: attentionMaskArray, token_type_ids: tokenTypeIdsArray)

        do {
            let prediction = try model.prediction(input: input)
            let modelOutput = (0..<prediction.Identity.count).map { prediction.Identity[$0].doubleValue }

            let embeddingDim = 384
            let reshapedOutput = stride(from: 0, to: modelOutput.count, by: embeddingDim).map {
                Array(modelOutput[$0..<min($0 + embeddingDim, modelOutput.count)])
            }

            let pooled = meanPooling(reshapedOutput, attentionMask: attentionMask.map { Double($0) })
            let normalized = normalizeEmbedding(pooled)

            DispatchQueue.main.async {
                if let document = pdfView?.document {
                    for pageIndex in 0..<document.pageCount {
                        if let page = document.page(at: pageIndex) {
                            let highlightsToRemove = page.annotations.filter { $0.type == "Highlight" }
                            for annotation in highlightsToRemove {
                                page.removeAnnotation(annotation)
                            }
                        }
                    }
                }

                let matches = self.findTopNSimilarEmbeddings(queryEmbedding: normalized, topN: 5)
                for match in matches {
                    let highlight = PDFAnnotation(bounds: match.position.bounds, forType: .highlight, withProperties: nil)
                    highlight.color = .yellow

                    if let page = pdfView?.document?.page(at: match.position.pageNumber) {
                        page.addAnnotation(highlight)
                    }
                }
                self.isHidden = true
                self.topMatches = matches
                print("done")
            }
        } catch {
            print("Failed to get prediction: \(error)")
        }
    }






      
    func findTopNSimilarEmbeddings(queryEmbedding: [Double], topN: Int) -> [EmbeddingWithPosition] {

        let floatQuery = queryEmbedding.map { Float($0) }
        var resultIds = [Int32](repeating: -1, count: topN)

        floatQuery.withUnsafeBufferPointer { buffer in
            resultIds.withUnsafeMutableBufferPointer { resultBuffer in
                knn_search_hnsw(hnsw, buffer.baseAddress, Int32(buffer.count), Int32(topN), resultBuffer.baseAddress)
            }
        }
        var matches: [EmbeddingWithPosition] = []

        for id in resultIds {
            if id == -1 { continue }
            let index = Int(id)
            if index < self.embeddingsWithPositions.count && index < pdfTextWithPosition.count {
                let emb = self.embeddingsWithPositions[index]
                matches.append(emb)
            } else {
            }
        }

        return matches
    }




    func fit(embeddings: [EmbeddingWithPosition]) {
        NSLog("Embeddings in ViewModel: \(self.embeddingsWithPositions.count)")
        for embedding in embeddings {
            let floatVec = embedding.embedding.map { Float($0) }
            floatVec.withUnsafeBufferPointer { buffer in
                insert_hnsw(hnsw, buffer.baseAddress, Int32(buffer.count), Int32(embedding.id))
            }
        }
    }


    

}


    
