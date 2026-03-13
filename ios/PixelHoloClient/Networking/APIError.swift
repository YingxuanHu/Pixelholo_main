import Foundation

enum APIError: LocalizedError {
    case invalidBaseURL
    case invalidResponse
    case server(statusCode: Int, message: String)
    case decoding(Error)
    case transport(Error)

    var errorDescription: String? {
        switch self {
        case .invalidBaseURL:
            return "Server URL is invalid."
        case .invalidResponse:
            return "Received an invalid response from server."
        case let .server(statusCode, message):
            return "Server error (\(statusCode)): \(message)"
        case let .decoding(error):
            return "Failed to decode server response: \(error.localizedDescription)"
        case let .transport(error):
            return "Network error: \(error.localizedDescription)"
        }
    }
}

