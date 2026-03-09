import Combine
import Foundation

@MainActor
final class ServerConfig: ObservableObject {
    static let shared = ServerConfig()

    @Published var baseURLString: String {
        didSet {
            UserDefaults.standard.set(baseURLString, forKey: Self.storageKey)
        }
    }

    private static let storageKey = "pixelholo.server.baseURL"

    private init() {
        self.baseURLString = UserDefaults.standard.string(forKey: Self.storageKey)
            ?? "http://127.0.0.1:8000"
    }

    var baseURL: URL? {
        let trimmed = baseURLString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return nil
        }

        let candidate: String
        if trimmed.hasPrefix("http://") || trimmed.hasPrefix("https://") {
            candidate = trimmed
        } else {
            candidate = "http://\(trimmed)"
        }

        return URL(string: candidate)
    }
}
