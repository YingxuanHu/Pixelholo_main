import Combine
import Foundation

@MainActor
final class ProfileListViewModel: ObservableObject {
    @Published private(set) var voiceProfiles: [ProfileInfo] = []
    @Published private(set) var avatarProfiles: [ProfileInfo] = []
    @Published private(set) var isLoading = false
    @Published var errorMessage: String?

    private let apiClient: APIClient

    init(apiClient: APIClient? = nil) {
        self.apiClient = apiClient ?? APIClient()
    }

    func loadProfiles(baseURL: URL?) async {
        guard let baseURL else {
            errorMessage = "Enter the backend URL first. Use your VM or LAN IP, not 127.0.0.1 unless the backend is running on this same Mac."
            voiceProfiles = []
            avatarProfiles = []
            return
        }

        isLoading = true
        errorMessage = nil
        defer { isLoading = false }

        do {
            let profiles = try await apiClient.fetchProfiles(baseURL: baseURL)
            voiceProfiles = profiles
                .filter { $0.profileType == .voice }
                .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
            avatarProfiles = profiles
                .filter { $0.profileType == .avatar }
                .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
        } catch {
            errorMessage = Self.errorMessage(for: error, baseURL: baseURL)
            voiceProfiles = []
            avatarProfiles = []
        }
    }

    private static func errorMessage(for error: Error, baseURL: URL) -> String {
        let baseMessage = error.localizedDescription
        guard usesLoopbackHost(baseURL) else {
            return baseMessage
        }
        return "\(baseMessage)\n127.0.0.1 only works when the backend is running on this same machine. For a VM or another computer, enter its reachable LAN IP instead."
    }

    private static func usesLoopbackHost(_ url: URL) -> Bool {
        guard let host = url.host?.lowercased() else {
            return false
        }
        return host == "127.0.0.1" || host == "localhost" || host == "::1"
    }
}
