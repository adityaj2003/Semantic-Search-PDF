import SwiftUI
import PDFKit
import CoreML
import AppKit

struct ContentView: View {
    @State private var searchText = ""
    @State private var embeddingsWithPositions: [EmbeddingWithPosition] = []
    @State private var pdfTextWithPosition: [TextWithPosition] = []
    @State private var pdfKitView: PDFKitView?
    @State private var topMatches: [EmbeddingWithPosition] = []
    @State private var selectedMatch: EmbeddingWithPosition?
    
        var body: some View {
            GeometryReader { geometry in
                NavigationSplitView {
                    List {
                        ForEach(topMatches, id: \.position.text) { match in
                            Button(action: {
                                selectedMatch = match
                                highlightSelectedMatch(match: match, pdfView: pdfKitView?.getView())
                            }) {
                                HStack {
                                    VStack(alignment: .leading) {
                                        Text(match.position.text)
                                            .font(.headline)
                                            .lineLimit(2)
                                        Text("Page \(match.position.pageNumber + 1)")
                                            .font(.subheadline)
                                    }
                                    Spacer()
                                }
                                .padding()
                            }
                            .buttonStyle(PlainButtonStyle())
                        }
                    }
                    .frame(width: geometry.size.width * 0.2)
                } detail: {
                    VStack {
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
                                        highlightTopMatches(pdfView: newPDFKitView.getView())
                                    }
                            }
                        } else {
                            Text("PDF not found")
                        }
                    }
                    .frame(width: geometry.size.width * 0.8) // Content area width
                }
                .toolbar {
                }
                .searchable(text: $searchText, prompt: "Search")
                .onSubmit(of: .search) {
                    fetchEmbeddings()
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
            var tokenIds = tokenizer.tokenizeToIds(text: chunk.text)
            var attentionMask = Array(repeating: 1, count: tokenIds.count)
            var tokenTypeIds = Array(repeating: 0, count: tokenIds.count)
            
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
                let embedding = (0..<prediction.Identity.count).map { prediction.Identity[$0].doubleValue }
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
                    if let selection = page.selection(for: NSRange(location: pageText.distance(from: pageText.startIndex, to: pageText.range(of: sentence)!.lowerBound), length: sentence.count)), selection.string!.count > 5 {
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
    
    func findTopNSimilarEmbeddings(queryEmbedding: [Double], sentenceEmbeddings: [EmbeddingWithPosition], topN: Int) -> [EmbeddingWithPosition] {
        let similarities = sentenceEmbeddings.map { embeddingWithPosition in
            return (embeddingWithPosition, cosineSimilarity(queryEmbedding, embeddingWithPosition.embedding))
        }
        let sortedEmbeddings = similarities.sorted { $0.1 > $1.1 }.prefix(topN)
        return sortedEmbeddings.map { $0.0 }
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
        
        let adjustedTokenIds = adjustTokenIds(tokenIds)
        let adjustedAttentionMask = Array(repeating: 1, count: adjustedTokenIds.count)
        let adjustedTokenTypeIds = Array(repeating: 0, count: adjustedTokenIds.count)
        
        guard let inputIdsArray = try? MLMultiArray(shape: [1, NSNumber(value: adjustedTokenIds.count)], dataType: .int32),
              let attentionMaskArray = try? MLMultiArray(shape: [1, NSNumber(value: adjustedTokenIds.count)], dataType: .int32),
              let tokenTypeIdsArray = try? MLMultiArray(shape: [1, NSNumber(value: adjustedTokenIds.count)], dataType: .int32) else {
            print("Failed to create MLMultiArray")
            return
        }
        
        for (index, tokenId) in adjustedTokenIds.enumerated() {
            inputIdsArray[index] = NSNumber(value: tokenId)
            attentionMaskArray[index] = NSNumber(value: adjustedAttentionMask[index])
            tokenTypeIdsArray[index] = NSNumber(value: adjustedTokenTypeIds[index])
        }
        
        let input = MiniLM_V6Input(input_ids: inputIdsArray, attention_mask: attentionMaskArray, token_type_ids: tokenTypeIdsArray)
        
        do {
            let prediction = try model.prediction(input: input)
            let queryEmbedding = (0..<prediction.Identity.count).map { prediction.Identity[$0].doubleValue }
            topMatches = findTopNSimilarEmbeddings(queryEmbedding: queryEmbedding, sentenceEmbeddings: embeddingsWithPositions, topN: 5)
            for match in topMatches {
                let highlight = PDFAnnotation(bounds: match.position.bounds, forType: .highlight, withProperties: nil)
                highlight.color = .yellow
                
                if let page = pdfView?.document?.page(at: match.position.pageNumber) {
                    page.addAnnotation(highlight)
                }
            }
        } catch {
            print("Failed to get prediction: \(error)")
        }
    }
    func highlightSelectedMatch(match: EmbeddingWithPosition, pdfView: PDFView?) {
        pdfView?.go(to: match.position.bounds, on: (pdfView?.document?.page(at: match.position.pageNumber))!)
    }
    

    func fetchEmbeddings() {
        if let document = pdfKitView?.getView().document {
            for i in 0..<document.pageCount {
                if let page = document.page(at: i) {
                    let annotations = page.annotations
                    for annotation in annotations {
                        page.removeAnnotation(annotation)
                    }
                }
            }
        }
        let tokenizer = BertTokenizer()
        
        guard let model = try? MiniLM_V6(configuration: .init()) else {
            print("Failed to load model")
            return
        }
        
        embeddingsWithPositions = tokenizeAndEmbed(textChunks: pdfTextWithPosition, tokenizer: tokenizer, model: model)
        highlightTopMatches(pdfView: pdfKitView?.getView())
    }
}
