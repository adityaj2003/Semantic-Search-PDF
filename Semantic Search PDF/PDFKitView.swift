import SwiftUI
import Foundation
import PDFKit

struct PDFKitView: NSViewRepresentable {
    let url: URL
    let pdfView : PDFView = PDFView()

    func makeNSView(context: Context) -> PDFView {
        pdfView.document = PDFDocument(url: url)
        pdfView.autoScales = true
        return pdfView
    }

    func updateNSView(_ pdfView: PDFView, context: Context) {
        if let document = PDFDocument(url: url) {
            pdfView.document = document
        }
        else {
        }
    }
    
    func getView() -> PDFView {
        return pdfView;
    }
}
