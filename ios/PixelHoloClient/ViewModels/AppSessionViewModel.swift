import Combine
import Foundation

@MainActor
final class AppSessionViewModel: ObservableObject {
    enum Tab: Hashable {
        case profiles
        case stream
        case pipeline
        case create
    }

    @Published var selectedTab: Tab = .profiles
    @Published var selectedProfile: ProfileInfo?

    func openProfileForStreaming(_ profile: ProfileInfo) {
        selectedProfile = profile
        selectedTab = .stream
    }

    func clearSelectedProfile() {
        selectedProfile = nil
    }
}
