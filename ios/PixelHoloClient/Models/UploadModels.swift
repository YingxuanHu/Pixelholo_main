import Foundation

struct UploadResponse: Codable, Hashable {
    let savedPath: String
    let filename: String

    enum CodingKeys: String, CodingKey {
        case savedPath = "saved_path"
        case filename
    }
}

struct InterruptResponse: Codable, Hashable {
    let status: String
    let interruptedStreams: Int

    enum CodingKeys: String, CodingKey {
        case status
        case interruptedStreams = "interrupted_streams"
    }
}

struct WarmupRequest: Codable, Hashable {
    let profile: String
    let profileType: ProfileType
    var lipsyncBackend: LipsyncBackend?
    var force: Bool = false

    enum CodingKeys: String, CodingKey {
        case profile
        case profileType = "profile_type"
        case lipsyncBackend = "lipsync_backend"
        case force
    }
}

struct WarmupResponse: Codable, Hashable {
    let status: String
    let profile: String
    let profileType: ProfileType
    let lipsyncBackend: String?

    enum CodingKeys: String, CodingKey {
        case status
        case profile
        case profileType = "profile_type"
        case lipsyncBackend = "lipsync_backend"
    }
}
