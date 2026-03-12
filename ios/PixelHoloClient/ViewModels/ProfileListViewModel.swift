import Combine
import Foundation

@MainActor
final class ProfileListViewModel: ObservableObject {
    enum LoadReason {
        case automatic
        case manual
    }

    @Published private(set) var voiceProfiles: [ProfileInfo] = []
    @Published private(set) var avatarProfiles: [ProfileInfo] = []
    @Published private(set) var isLoading = false
    @Published var errorMessage: String?

    private let apiClient: APIClient
    private var lastLoadedBaseURLString: String?
    private var activeLoadToken = UUID()

    init(apiClient: APIClient? = nil) {
        self.apiClient = apiClient ?? APIClient()
    }

    func loadProfiles(baseURL: URL?, reason: LoadReason = .manual) async {
        guard let baseURL else {
            errorMessage = "Enter the backend URL first. Use your VM or LAN IP, not 127.0.0.1 unless the backend is running on this same Mac."
            voiceProfiles = []
            avatarProfiles = []
            lastLoadedBaseURLString = nil
            return
        }

        if reason == .automatic, lastLoadedBaseURLString == baseURL.absoluteString, !voiceProfiles.isEmpty || !avatarProfiles.isEmpty {
            return
        }

        if isLoading {
            return
        }

        let loadToken = UUID()
        activeLoadToken = loadToken
        isLoading = true
        errorMessage = nil
        defer { isLoading = false }

        do {
            let profiles = try await apiClient.fetchProfiles(baseURL: baseURL)
            guard activeLoadToken == loadToken, !Task.isCancelled else {
                return
            }

            voiceProfiles = profiles
                .filter { $0.profileType == .voice }
                .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
            avatarProfiles = profiles
                .filter { $0.profileType == .avatar }
                .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
            lastLoadedBaseURLString = baseURL.absoluteString
        } catch {
            guard activeLoadToken == loadToken else {
                return
            }

            if Self.isCancellation(error) || Task.isCancelled {
                return
            }
            errorMessage = Self.errorMessage(for: error, baseURL: baseURL)
            voiceProfiles = []
            avatarProfiles = []
            lastLoadedBaseURLString = nil
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

    private static func isCancellation(_ error: Error) -> Bool {
        if error is CancellationError {
            return true
        }

        let nsError = error as NSError
        if nsError.domain == NSURLErrorDomain, nsError.code == NSURLErrorCancelled {
            return true
        }

        if case let APIError.transport(transportError) = error {
            return isCancellation(transportError)
        }

        return false
    }
}
