
import SwiftUI
import PDFKit
import CoreML


struct ContentView: View {
    @State private var searchText = ""
    @State private var embeddingsWithPositions: [EmbeddingWithPosition] = []
    @State private var pdfTextWithPosition: [TextWithPosition] = []
        
    var body: some View {
            VStack {
                TextField("Search", text: $searchText, onCommit: fetchEmbeddings)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .padding()
                
                if let downloadsDirectory = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first {
                    let pdfURL = downloadsDirectory.appendingPathComponent("Aditya_Resume.pdf")
                    
                    PDFKitView(url: pdfURL)
                        .edgesIgnoringSafeArea(.all)
                        .onAppear {
                            pdfTextWithPosition = extractTextWithPosition(from: pdfURL)
                        }
                } else {
                    Text("PDF not found")
                }
                
                if let bestMatch = findBestMatch() {
                    Text("Best Match: \(bestMatch.position.text) on page \(bestMatch.position.pageNumber + 1)")
                }
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
            
            let inputIdsArray = try? MLMultiArray(shape: [1, NSNumber(value: tokenIds.count)], dataType: .int32)
            let attentionMaskArray = try? MLMultiArray(shape: [1, NSNumber(value: tokenIds.count)], dataType: .int32)
            let tokenTypeIdsArray = try? MLMultiArray(shape: [1, NSNumber(value: tokenIds.count)], dataType: .int32)
            
            for (index, tokenId) in tokenIds.enumerated() {
                inputIdsArray?[index] = NSNumber(value: tokenId)
                attentionMaskArray?[index] = NSNumber(value: attentionMask[index])
                tokenTypeIdsArray?[index] = NSNumber(value: tokenTypeIds[index])
            }
            
            let input = MiniLM_V6Input(attention_mask: attentionMaskArray!, input_ids: inputIdsArray!, token_type_ids: tokenTypeIdsArray!)
            
            if let prediction = try? model.prediction(input: input) {
                let embedding = (0..<prediction.Identity.count).map { prediction.Identity[$0].doubleValue }
                let embeddingWithPosition = EmbeddingWithPosition(embedding: embedding, position: chunk)
                embeddingsWithPositions.append(embeddingWithPosition)
            }
        }
        
        return embeddingsWithPositions
    }
    
    
    func extractTextWithPosition(from url: URL) -> [TextWithPosition] {
        guard let pdfDocument = PDFDocument(url: url) else { return [] }
        var textWithPositions: [TextWithPosition] = []
        
        for pageIndex in 0..<pdfDocument.pageCount {
            guard let page = pdfDocument.page(at: pageIndex) else { continue }
            
            let pageBounds = page.bounds(for: .mediaBox)
            
            if let pageText = page.string {
                let sentences = pageText.split(separator: ".").map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
                for sentence in sentences {
                    let position = TextWithPosition(
                        text: sentence,
                        pageNumber: pageIndex,
                        bounds: pageBounds
                    )
                    textWithPositions.append(position)
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

    func findMostRelevantSentence(queryEmbedding: [Double], sentenceEmbeddings: [EmbeddingWithPosition]) -> EmbeddingWithPosition? {
        var maxSimilarity = -1.0
        var bestMatch: EmbeddingWithPosition? = nil
        for embeddingWithPosition in sentenceEmbeddings {
            let similarity = cosineSimilarity(queryEmbedding, embeddingWithPosition.embedding)
            if similarity > maxSimilarity {
                maxSimilarity = similarity
                bestMatch = embeddingWithPosition
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
    func findBestMatch() -> EmbeddingWithPosition? {
            let tokenizer = BertTokenizer()
            guard let model = try? MiniLM_V6(configuration: .init()) else {
                print("Failed to load model")
                return nil
            }
            
            let tokenIds = tokenizer.tokenizeToIds(text: searchText)
            let attentionMask = Array(repeating: 1, count: tokenIds.count)
            let tokenTypeIds = Array(repeating: 0, count: tokenIds.count)
            
            let inputIdsArray = try? MLMultiArray(shape: [1, NSNumber(value: tokenIds.count)], dataType: .int32)
            let attentionMaskArray = try? MLMultiArray(shape: [1, NSNumber(value: tokenIds.count)], dataType: .int32)
            let tokenTypeIdsArray = try? MLMultiArray(shape: [1, NSNumber(value: tokenIds.count)], dataType: .int32)
            
            for (index, tokenId) in tokenIds.enumerated() {
                inputIdsArray?[index] = NSNumber(value: tokenId)
                attentionMaskArray?[index] = NSNumber(value: attentionMask[index])
                tokenTypeIdsArray?[index] = NSNumber(value: tokenTypeIds[index])
            }
            
            let input = MiniLM_V6Input(attention_mask: attentionMaskArray!, input_ids: inputIdsArray!, token_type_ids: tokenTypeIdsArray!)
            
            if let prediction = try? model.prediction(input: input) {
                let queryEmbedding = (0..<prediction.Identity.count).map { prediction.Identity[$0].doubleValue }
                return findMostRelevantSentence(queryEmbedding: queryEmbedding, sentenceEmbeddings: embeddingsWithPositions)
            } else {
                print("Failed to get prediction")
                return nil
            }
        }
    
    
}


