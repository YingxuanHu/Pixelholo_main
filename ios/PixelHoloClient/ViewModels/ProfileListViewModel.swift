import Foundation

@MainActor
final class ProfileListViewModel: ObservableObject {
    @Published private(set) var voiceProfiles: [ProfileInfo] = []
    @Published private(set) var avatarProfiles: [ProfileInfo] = []
    @Published private(set) var isLoading = false
    @Published var errorMessage: String?

    private let apiClient: APIClient

    init(apiClient: APIClient = APIClient()) {
        self.apiClient = apiClient
    }

    func loadProfiles(baseURL: URL?) async {
        guard let baseURL else {
            errorMessage = APIError.invalidBaseURL.localizedDescription
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
            errorMessage = error.localizedDescription
            voiceProfiles = []
            avatarProfiles = []
        }
    }
}

