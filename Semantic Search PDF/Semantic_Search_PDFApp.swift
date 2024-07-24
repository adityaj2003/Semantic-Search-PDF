import PDFKit
import CoreML
//  Created by Aditya Jadhav on 09/07/24.
//

import SwiftUI

@available(macOS 14.0, *)

@main
struct Semantic_Search_PDFApp: App {
    @StateObject private var viewModel = ViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView(viewModel: viewModel)
        }.commands {
            CommandGroup(replacing: CommandGroupPlacement.newItem) {
                Button("Open File...") {
                    viewModel.isFileImporterPresented.toggle()
                }
                .keyboardShortcut("O")
            }
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
