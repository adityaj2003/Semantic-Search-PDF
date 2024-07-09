import SwiftUI
import Foundation
import PDFKit

struct PDFKitView: NSViewRepresentable {
    let url: URL

    func makeNSView(context: Context) -> PDFView {
        let pdfView = PDFView()
        pdfView.autoScales = true
        return pdfView
    }

    func updateNSView(_ pdfView: PDFView, context: Context) {
        print(url)
        if let document = PDFDocument(url: url) {
            pdfView.document = document
        }
        else {
            print("Nothing")
        }
    }
}
