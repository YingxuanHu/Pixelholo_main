import SwiftUI

@main
@MainActor
struct PixelHoloClientApp: App {
    @StateObject private var serverConfig = ServerConfig.shared

    var body: some Scene {
        WindowGroup {
            NavigationStack {
                ProfileListView()
            }
            .environmentObject(serverConfig)
        }
    }
}

