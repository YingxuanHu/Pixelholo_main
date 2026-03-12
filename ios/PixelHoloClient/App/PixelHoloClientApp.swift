import SwiftUI

@main
@MainActor
struct PixelHoloClientApp: App {
    @StateObject private var serverConfig = ServerConfig.shared
    @StateObject private var appSession = AppSessionViewModel()

    var body: some Scene {
        WindowGroup {
            TabView(selection: $appSession.selectedTab) {
                NavigationStack {
                    ProfileListView()
                }
                .tabItem {
                    Label("Profiles", systemImage: "person.3")
                }
                .tag(AppSessionViewModel.Tab.profiles)

                NavigationStack {
                    AvatarChatView()
                }
                .tabItem {
                    Label("Stream", systemImage: "waveform")
                }
                .tag(AppSessionViewModel.Tab.stream)

                NavigationStack {
                    PipelineControlView()
                }
                .tabItem {
                    Label("Train", systemImage: "gearshape.2")
                }
                .tag(AppSessionViewModel.Tab.pipeline)

                NavigationStack {
                    CreateProfileView()
                }
                .tabItem {
                    Label("Create", systemImage: "plus.circle")
                }
                .tag(AppSessionViewModel.Tab.create)
            }
            .toolbarBackground(.visible, for: .tabBar)
            .toolbarBackground(Color(uiColor: .systemBackground), for: .tabBar)
            .environmentObject(serverConfig)
            .environmentObject(appSession)
            .environmentObject(appSession.streamSession)
        }
    }
}
