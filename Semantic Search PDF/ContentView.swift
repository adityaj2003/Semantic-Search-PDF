import SwiftUI
import PDFKit
import CoreML

struct ContentView: View {
    @State private var searchText = "save data for timestamp"
    @State private var embeddingsWithPositions: [EmbeddingWithPosition] = []
    @State private var pdfTextWithPosition: [TextWithPosition] = []
    @State private var pdfKitView: PDFKitView?

    var body: some View {
        VStack {
            TextField("Search", text: $searchText, onCommit: fetchEmbeddings)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding()

            if let downloadsDirectory = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first {
                let pdfURL = downloadsDirectory.appendingPathComponent("ADC_GUI_Docs.pdf")

                if let pdfKitView = pdfKitView {
                    pdfKitView
                        .edgesIgnoringSafeArea(.all)
                        .onAppear {
                            pdfTextWithPosition = extractTextWithPosition(from: pdfURL)
                        }
                } else {
                    PDFKitView(url: pdfURL)
                        .edgesIgnoringSafeArea(.all)
                        .onAppear {
                            let newPDFKitView = PDFKitView(url: pdfURL)
                            pdfKitView = newPDFKitView
                            pdfTextWithPosition = extractTextWithPosition(from: pdfURL)

                            if let bestMatch = findBestMatch(pdfView: newPDFKitView.getView()) {
                                let highlight = PDFAnnotation(bounds: bestMatch.position.bounds, forType: .highlight, withProperties: nil)
                                highlight.color = .yellow

                                if let page = newPDFKitView.getView().document?.page(at: bestMatch.position.pageNumber) {
                                    page.addAnnotation(highlight)
                                }
                            }
                        }
                }
            } else {
                Text("PDF not found")
            }

            let bestMatch = findBestMatch(pdfView: pdfKitView?.getView())
        }
    }

    struct TextWithPosition {
        let text: String
        let pageNumber: Int
        let bounds: CGRect
    }

    struct EmbeddingWithPosition {
        let embedding: [Double]
        let position: TextWithPosition
    }

    func tokenizeAndEmbed(textChunks: [TextWithPosition], tokenizer: BertTokenizer, model: MiniLM_V6) -> [EmbeddingWithPosition] {
        var embeddingsWithPositions: [EmbeddingWithPosition] = []

        for chunk in textChunks {
            let tokenIds = tokenizer.tokenizeToIds(text: chunk.text)
            let attentionMask = Array(repeating: 1, count: tokenIds.count)
            let tokenTypeIds = Array(repeating: 0, count: tokenIds.count)

            // Ensure tokenIds count is within the expected range, if not, adjust accordingly
            let adjustedTokenIds = adjustTokenIds(tokenIds)
            let adjustedAttentionMask = Array(repeating: 1, count: adjustedTokenIds.count)
            let adjustedTokenTypeIds = Array(repeating: 0, count: adjustedTokenIds.count)

            guard let inputIdsArray = try? MLMultiArray(shape: [1, NSNumber(value: adjustedTokenIds.count)], dataType: .int32),
                  let attentionMaskArray = try? MLMultiArray(shape: [1, NSNumber(value: adjustedTokenIds.count)], dataType: .int32),
                  let tokenTypeIdsArray = try? MLMultiArray(shape: [1, NSNumber(value: adjustedTokenIds.count)], dataType: .int32) else {
                print("Failed to create MLMultiArray")
                continue
            }

            for (index, tokenId) in adjustedTokenIds.enumerated() {
                inputIdsArray[index] = NSNumber(value: tokenId)
                attentionMaskArray[index] = NSNumber(value: adjustedAttentionMask[index])
                tokenTypeIdsArray[index] = NSNumber(value: adjustedTokenTypeIds[index])
            }

            let input = MiniLM_V6Input(attention_mask: attentionMaskArray, input_ids: inputIdsArray, token_type_ids: tokenTypeIdsArray)

            do {
                let prediction = try model.prediction(input: input)
                let embedding = (0..<prediction.Identity.count).map { prediction.Identity[$0].doubleValue }
                
                // Normalize embeddings
                let normalizedEmbedding = normalizeEmbedding(embedding)

                let embeddingWithPosition = EmbeddingWithPosition(embedding: normalizedEmbedding, position: chunk)
                embeddingsWithPositions.append(embeddingWithPosition)
            } catch {
                print("Failed to get Embeddings: \(error)")
            }
        }
        return embeddingsWithPositions
    }

    func adjustTokenIds(_ tokenIds: [Int]) -> [Int] {
        if tokenIds.count < 1 {
            return Array(tokenIds.prefix(1))
        } else if tokenIds.count > 2 {
            return Array(tokenIds.prefix(2))
        }
        return tokenIds
    }

    func normalizeEmbedding(_ embedding: [Double]) -> [Double] {
        let norm = sqrt(embedding.reduce(0) { $0 + $1 * $1 })
        return embedding.map { $0 / norm }
    }

    func extractTextWithPosition(from url: URL) -> [TextWithPosition] {
        guard let pdfDocument = PDFDocument(url: url) else { return [] }
        var textWithPositions: [TextWithPosition] = []

        for pageIndex in 0..<pdfDocument.pageCount {
            guard let page = pdfDocument.page(at: pageIndex) else { continue }

            if let pageText = page.string {
                let sentences = pageText.split(separator: ".").map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
                for sentence in sentences {
                    if let selection = page.selection(for: NSRange(location: pageText.distance(from: pageText.startIndex, to: pageText.range(of: sentence)!.lowerBound), length: sentence.count)) {
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

        return textWithPositions
    }

    func cosineSimilarity(_ vectorA: [Double], _ vectorB: [Double]) -> Double {
        let dotProduct = zip(vectorA, vectorB).map(*).reduce(0, +)
        let magnitudeA = sqrt(vectorA.map { $0 * $0 }.reduce(0, +))
        let magnitudeB = sqrt(vectorB.map { $0 * $0 }.reduce(0, +))
        return dotProduct / (magnitudeA * magnitudeB)
    }

    func findMostRelevantSentence(queryEmbedding: [Double], sentenceEmbeddings: [EmbeddingWithPosition], pdfView: PDFView?) -> EmbeddingWithPosition? {
        var maxSimilarity = -1.0
        var bestMatch: EmbeddingWithPosition? = nil
        for embeddingWithPosition in sentenceEmbeddings {
            let similarity = cosineSimilarity(queryEmbedding, embeddingWithPosition.embedding)
            if similarity > maxSimilarity {
                maxSimilarity = similarity
                bestMatch = embeddingWithPosition
            }
        }

        if let bestMatch = bestMatch, let pdfView = pdfView {
            let highlight = PDFAnnotation(bounds: bestMatch.position.bounds, forType: .highlight, withProperties: nil)
            highlight.color = .yellow

            if let page = pdfView.document?.page(at: bestMatch.position.pageNumber) {
                page.addAnnotation(highlight)
            }
        }
        return bestMatch
    }

    func fetchEmbeddings() {
        let tokenizer = BertTokenizer()

        guard let model = try? MiniLM_V6(configuration: .init()) else {
            print("Failed to load model")
            return
        }

        embeddingsWithPositions = tokenizeAndEmbed(textChunks: pdfTextWithPosition, tokenizer: tokenizer, model: model)
    }

    func findBestMatch(pdfView: PDFView?) -> EmbeddingWithPosition? {
        let tokenizer = BertTokenizer()
        guard let model = try? MiniLM_V6(configuration: .init()) else {
            print("Failed to load model")
            return nil
        }

        let tokenIds = tokenizer.tokenizeToIds(text: searchText)
        let attentionMask = Array(repeating: 1, count: tokenIds.count)
        let tokenTypeIds = Array(repeating: 0, count: tokenIds.count)

        // Ensure tokenIds count is within the expected range
        let adjustedTokenIds = adjustTokenIds(tokenIds)
        let adjustedAttentionMask = Array(repeating: 1, count: adjustedTokenIds.count)
        let adjustedTokenTypeIds = Array(repeating: 0, count: adjustedTokenIds.count)

        guard let inputIdsArray = try? MLMultiArray(shape: [1, NSNumber(value: adjustedTokenIds.count)], dataType: .int32),
              let attentionMaskArray = try? MLMultiArray(shape: [1, NSNumber(value: adjustedTokenIds.count)], dataType: .int32),
              let tokenTypeIdsArray = try? MLMultiArray(shape: [1, NSNumber(value: adjustedTokenIds.count)], dataType: .int32) else {
            print("Failed to create MLMultiArray")
            return nil
        }

        for (index, tokenId) in adjustedTokenIds.enumerated() {
            inputIdsArray[index] = NSNumber(value: tokenId)
            attentionMaskArray[index] = NSNumber(value: adjustedAttentionMask[index])
            tokenTypeIdsArray[index] = NSNumber(value: adjustedTokenTypeIds[index])
        }

        let input = MiniLM_V6Input(attention_mask: attentionMaskArray, input_ids: inputIdsArray, token_type_ids: tokenTypeIdsArray)

        do {
            let prediction = try model.prediction(input: input)
            let queryEmbedding = (0..<prediction.Identity.count).map { prediction.Identity[$0].doubleValue }
            return findMostRelevantSentence(queryEmbedding: queryEmbedding, sentenceEmbeddings: embeddingsWithPositions, pdfView: pdfView)
        } catch {
            print("Failed to get prediction: \(error)")
            return nil
        }
    }
}

