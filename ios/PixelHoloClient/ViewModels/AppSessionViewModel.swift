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
    let streamSession: AvatarChatViewModel
    @Published var stagedProfileName: String?
    @Published var stagedProfileType: ProfileType?

    init(streamSession: AvatarChatViewModel? = nil) {
        self.streamSession = streamSession ?? AvatarChatViewModel()
    }

    func openProfileForStreaming(_ profile: ProfileInfo) {
        selectedProfile = profile
        stagedProfileName = profile.name
        stagedProfileType = profile.profileType
        streamSession.applySelectedProfile(profile)
        selectedTab = .stream
    }

    func clearSelectedProfile() {
        selectedProfile = nil
    }

    func stageProfileWorkflow(name: String, type: ProfileType) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        stagedProfileName = trimmed
        stagedProfileType = type
    }
}
