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
    let runtimeInstanceID: String?
    let ttsHotBefore: Bool?
    let ttsWarmed: Bool?
    let lipsyncHotBefore: Bool?
    let lipsyncWarmed: Bool?
    let elapsedMS: Double?

    enum CodingKeys: String, CodingKey {
        case status
        case profile
        case profileType = "profile_type"
        case lipsyncBackend = "lipsync_backend"
        case runtimeInstanceID = "runtime_instance_id"
        case ttsHotBefore = "tts_hot_before"
        case ttsWarmed = "tts_warmed"
        case lipsyncHotBefore = "lipsync_hot_before"
        case lipsyncWarmed = "lipsync_warmed"
        case elapsedMS = "elapsed_ms"
    }
}
