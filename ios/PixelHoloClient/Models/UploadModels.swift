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
    var includeLLM: Bool = false
    var llmMode: LLMMode?
    var llmModel: String?
    var mobileProfile: Bool = false
    var avatarFPS: Double?
    var avatarMaxFrameEdge: Int?
    var musetalkInferFPS: Double?
    var musetalkStreamWindowSec: Double?
    var musetalkLookaheadSec: Double?

    enum CodingKeys: String, CodingKey {
        case profile
        case profileType = "profile_type"
        case lipsyncBackend = "lipsync_backend"
        case force
        case includeLLM = "include_llm"
        case llmMode = "llm_mode"
        case llmModel = "llm_model"
        case mobileProfile = "mobile_profile"
        case avatarFPS = "avatar_fps"
        case avatarMaxFrameEdge = "avatar_max_frame_edge"
        case musetalkInferFPS = "musetalk_infer_fps"
        case musetalkStreamWindowSec = "musetalk_stream_window_sec"
        case musetalkLookaheadSec = "musetalk_lookahead_sec"
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
    let ttsReady: Bool?
    let lipsyncHotBefore: Bool?
    let lipsyncWarmed: Bool?
    let lipsyncReady: Bool?
    let llmHotBefore: Bool?
    let llmWarmed: Bool?
    let llmReady: Bool?
    let elapsedMS: Double?

    enum CodingKeys: String, CodingKey {
        case status
        case profile
        case profileType = "profile_type"
        case lipsyncBackend = "lipsync_backend"
        case runtimeInstanceID = "runtime_instance_id"
        case ttsHotBefore = "tts_hot_before"
        case ttsWarmed = "tts_warmed"
        case ttsReady = "tts_ready"
        case lipsyncHotBefore = "lipsync_hot_before"
        case lipsyncWarmed = "lipsync_warmed"
        case lipsyncReady = "lipsync_ready"
        case llmHotBefore = "llm_hot_before"
        case llmWarmed = "llm_warmed"
        case llmReady = "llm_ready"
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
