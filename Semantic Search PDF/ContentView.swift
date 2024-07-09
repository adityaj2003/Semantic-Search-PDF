
import SwiftUI

struct ContentView: View {
    @State private var searchText = ""
    var body: some View {
        HStack {
            VStack {
                Text("Searching for \(searchText)")
                                    .navigationTitle("Searchable Example")
                                    .searchable(text: $searchText)
                
                if let downloadsDirectory = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first {
                    let pdfURL = downloadsDirectory.appendingPathComponent("Aditya_Resume.pdf")
                    
                    PDFKitView(url: pdfURL)
                        .edgesIgnoringSafeArea(.all)
                } else {
                    Text("PDF not found")
                }
            }
            .padding()
        }
    }
}


