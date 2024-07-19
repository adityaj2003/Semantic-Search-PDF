import PDFKit
import CoreML
//  Created by Aditya Jadhav on 09/07/24.
//

import SwiftUI

@available(macOS 14.0, *)
@main
struct Semantic_Search_PDFApp: App {
    @State private var isFileImporterPresented = true

    var body: some Scene {
        WindowGroup {
            ContentView(isFileImporterPresented: $isFileImporterPresented)
        }.commands {
            CommandGroup(replacing: CommandGroupPlacement.newItem) {Button("Open File...") {
                isFileImporterPresented.toggle()
            }
            .keyboardShortcut("O")}
        }
        
    }
}


class ModelHandler {
    private var model: MiniLM_V6

    init() {
        do {
            model = try MiniLM_V6(configuration: .init())
        } catch {
            fatalError("Failed to load model: \(error)")
        }
    }
    
}
