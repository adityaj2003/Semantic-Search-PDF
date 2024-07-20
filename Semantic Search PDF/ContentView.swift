import SwiftUI
import PDFKit
import CoreML
import AppKit

extension View {
    @ViewBuilder func isHidden(_ isHidden: Bool) -> some View {
        if isHidden {
            self.hidden()
        } else {
            self
        }
    }
}
@available(macOS 14.0, *)
struct ContentView: View {
    @State var selectedFileURL: URL?
    @Binding var isFileImporterPresented: Bool
    @State private var searchText = ""
    @State private var embeddingsWithPositions: [EmbeddingWithPosition] = []
    @State private var pdfTextWithPosition: [TextWithPosition] = []
    @State private var pdfKitView: PDFKitView?
    @State private var topMatches: [EmbeddingWithPosition] = []
    @State private var selectedMatch: EmbeddingWithPosition?
    @State private var isSearching = false
    @State var isHidden : Bool = true
    @State var isHiddenText : Bool = false
    @State var indexedPages : Int = 0
    init(isFileImporterPresented: Binding<Bool>) {
        _isFileImporterPresented = isFileImporterPresented
    }
        var body: some View {
            GeometryReader { geometry in
                NavigationSplitView {
                    Text("Pages Indexed: \(indexedPages)").isHidden(isHiddenText)
                    ProgressView().isHidden(isHidden)
                    List {
                        ForEach(topMatches, id: \.position.text) { match in
                            Button(action: {
                                selectedMatch = match
                                highlightSelectedMatch(match: match, pdfView: pdfKitView?.getView())
                            }) {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Page \(match.position.pageNumber + 1)")
                                        .font(.subheadline)
                                        .foregroundColor(.gray)
                                    Text(match.position.text)
                                        .font(.headline)
                                        .lineLimit(2)
                                }
                                .padding()
                                .cornerRadius(8)
                            }
                            .buttonStyle(PlainButtonStyle())
                            Divider()
                        }
                    }
                    .frame(width: geometry.size.width * 0.2)
                } detail: {
                    VStack {
                        if let pdfURL = selectedFileURL {
                            if let pdfKitView = pdfKitView {
                                pdfKitView
                                    .edgesIgnoringSafeArea(.all)
                            } else {
                                PDFKitView(url: pdfURL)
                                    .edgesIgnoringSafeArea(.all)
                            }
                        } else {
                            Text("PDF not found")
                        }
                    }
                    .frame(width: geometry.size.width * 0.8) // Content area width
                }
                .toolbar {
                    
                }
                
                .searchable(text: $searchText, isPresented: $isSearching, prompt: "Search")
                .background(Button("", action: { isSearching = true }).keyboardShortcut("f").hidden())
                .onSubmit(of: .search) {
                    isHidden = false
                    DispatchQueue.main.async {
                            highlightTopMatches(pdfView: pdfKitView?.getView())
                    }
                }.fileImporter(isPresented: $isFileImporterPresented, allowedContentTypes: [.pdf]) { result in
                    switch result {
                    case .success(let url):
                        selectedFileURL = url
                    case .failure(let error):
                        print("Error selecting file: \(error.localizedDescription)")
                    }
                }
                .onChange(of: selectedFileURL) { newURL in
                                if let pdfURL = newURL {
                                    indexedPages = 0
                                    pdfKitView = PDFKitView(url: pdfURL)
                                    DispatchQueue.global(qos: .userInteractive).async{
                           
                                        highlightTopMatches(pdfView: pdfKitView?.getView())
                                        embeddingsWithPositions = []
                                        pdfTextWithPosition = []
                                        topMatches = []
                                        isHiddenText = false
                                        pdfTextWithPosition = extractTextWithPosition(from:  pdfURL)
                                    }
                                }
                    
                    
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
            indexedPages = chunk.pageNumber
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

    // Example usage in your extractTextWithPosition function
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
        let tokenizer = BertTokenizer()
        guard let model = try? MiniLM_V6(configuration: .init()) else {
            print("Failed to load model")
            return textWithPositions
        }
        embeddingsWithPositions = tokenizeAndEmbed(textChunks: textWithPositions, tokenizer: tokenizer, model: model)
        isHiddenText = true
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
            let queryEmbedding = (0..<prediction.Identity.count).map { prediction.Identity[$0].doubleValue }
            let matches = findTopNSimilarEmbeddings(queryEmbedding: queryEmbedding, sentenceEmbeddings: embeddingsWithPositions, topN: 5)
            for match in matches {
                let highlight = PDFAnnotation(bounds: match.position.bounds, forType: .highlight, withProperties: nil)
                highlight.color = .yellow
                
                if let page = pdfView?.document?.page(at: match.position.pageNumber) {
                    page.addAnnotation(highlight)
                }
            }
            isHidden = true
            topMatches = matches
        } catch {
            print("Failed to get prediction: \(error)")
        }
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
    

}
