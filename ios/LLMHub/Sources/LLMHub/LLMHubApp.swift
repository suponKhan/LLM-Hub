import Foundation
import SwiftUI

@main
struct LLMHubApp: App {
    @StateObject private var settings = AppSettings.shared

    init() {
        let line = "[LLMHub] App launched\n"
        if let data = line.data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
        NSLog("[LLMHub] App launched")
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(settings)
                .preferredColorScheme(settings.theme.colorScheme)
                .environment(\.locale, settings.selectedLanguage.locale)
                .environment(\.layoutDirection, settings.selectedLanguage.isRTL ? .rightToLeft : .leftToRight)
        }
    }
}
