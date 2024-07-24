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
    @ObservedObject var viewModel: ViewModel
    @State private var selectedMatch: EmbeddingWithPosition?

    var body: some View {
        GeometryReader { geometry in
            NavigationSplitView {
                Text("Pages Indexed: \(viewModel.indexedPages)").isHidden(viewModel.isHiddenText)
                ProgressView().isHidden(viewModel.isHidden)
                List {
                    ForEach(viewModel.topMatches, id: \.position.text) { match in
                        Button(action: {
                            selectedMatch = match
                            if let pdfKitView = viewModel.pdfKitView {
                                viewModel.highlightSelectedMatch(match: match, pdfView: pdfKitView.getView())
                            }
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
                    if let pdfURL = viewModel.selectedFileURL {
                        if let pdfKitView = viewModel.pdfKitView {
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
                .frame(width: geometry.size.width * 0.8)
            }
            .toolbar {
                
            }
            .searchable(text: $viewModel.searchText, isPresented: $viewModel.isSearching, prompt: "Search")
            .background(Button("", action: { viewModel.isSearching = true }).keyboardShortcut("f").hidden())
            .onSubmit(of: .search) {
                viewModel.isHidden = false
                DispatchQueue.main.async {
                    viewModel.highlightTopMatches(pdfView: viewModel.pdfKitView?.getView())
                }
            }.fileImporter(isPresented: $viewModel.isFileImporterPresented, allowedContentTypes: [.pdf]) { result in
                viewModel.handleFileImport(result: result)
            }
            .onChange(of: viewModel.selectedFileURL) { newURL in
                viewModel.handleFileChange(newURL: newURL)
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

