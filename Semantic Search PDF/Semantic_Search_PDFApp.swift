import PDFKit
import CoreML
//  Created by Aditya Jadhav on 09/07/24.
//

import SwiftUI

@main
struct Semantic_Search_PDFApp: App {
    
    var body: some Scene {
        WindowGroup {
            ContentView()
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
