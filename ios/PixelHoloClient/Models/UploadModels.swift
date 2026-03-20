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
    let llmHotBefore: Bool?
    let llmWarmed: Bool?
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
        case llmHotBefore = "llm_hot_before"
        case llmWarmed = "llm_warmed"
        case elapsedMS = "elapsed_ms"
    }
}

struct LipsyncBackendStatusResponse: Codable, Hashable {
    let backend: String
    let runtimeInstanceID: String?

    enum CodingKeys: String, CodingKey {
        case backend
        case runtimeInstanceID = "runtime_instance_id"
    }
}

enum VoiceControlsLoadState: Equatable {
    case idle
    case loading
    case ready
    case unavailable(message: String)
    case error(message: String)
}

struct VoiceControlBackendDefaults: Hashable {
    let pitchShift: Double
    let f0Scale: Double
    let embeddingScale: Double
    let paceScale: Double
    let volumeGain: Double
}

struct VoiceControlValues: Hashable {
    var pitch: Double
    var pace: Int
    var tone: Int
    var volume: Int
}

struct ProfileVoiceControlsResponse: Codable, Hashable {
    struct Controls: Codable, Hashable {
        let pitchShift: Double?
        let f0Scale: Double?
        let embeddingScale: Double?
        let paceScale: Double?
        let volumeGain: Double?

        enum CodingKeys: String, CodingKey {
            case pitchShift = "pitch_shift"
            case f0Scale = "f0_scale"
            case embeddingScale = "embedding_scale"
            case paceScale = "pace_scale"
            case volumeGain = "volume_gain"
        }
    }

    let profile: String
    let profileType: ProfileType
    let controls: Controls

    enum CodingKeys: String, CodingKey {
        case profile
        case profileType = "profile_type"
        case controls
    }
}
