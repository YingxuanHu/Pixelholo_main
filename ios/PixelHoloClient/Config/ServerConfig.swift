import Combine
import Foundation

@MainActor
final class ServerConfig: ObservableObject {
    static let shared = ServerConfig()
    private static let defaultBaseURLString = "http://100.120.224.119:8000"

    @Published var baseURLString: String {
        didSet {
            UserDefaults.standard.set(baseURLString, forKey: Self.storageKey)
        }
    }

    private static let storageKey = "pixelholo.server.baseURL"

    private init() {
        let storedValue = UserDefaults.standard.string(forKey: Self.storageKey)?
            .trimmingCharacters(in: .whitespacesAndNewlines)

        if let storedValue, !storedValue.isEmpty {
            self.baseURLString = Self.isLoopbackAddress(storedValue)
                ? Self.defaultBaseURLString
                : storedValue
        } else {
            self.baseURLString = Self.defaultBaseURLString
        }
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

    var usesLoopbackHost: Bool {
        guard let rawValue = baseURL?.absoluteString else {
            return false
        }
        return Self.isLoopbackAddress(rawValue)
    }

    static func isLoopbackAddress(_ value: String) -> Bool {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        let candidate = trimmed.hasPrefix("http://") || trimmed.hasPrefix("https://")
            ? trimmed
            : "http://\(trimmed)"
        guard let host = URL(string: candidate)?.host?.lowercased() else {
            return false
        }
        return host == "127.0.0.1" || host == "localhost" || host == "::1"
    }
}
