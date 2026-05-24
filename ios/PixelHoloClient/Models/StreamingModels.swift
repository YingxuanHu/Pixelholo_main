import AVFoundation
import Foundation
import UIKit

enum StreamEndpoint: String, CaseIterable {
    case chat
    case speak
}

enum LipsyncBackend: String, Codable, CaseIterable {
    case wav2lip
    case musetalk
}

enum LLMMode: String, Codable, CaseIterable, Identifiable {
    case legacyFast = "legacy_fast"
    case liveSearch = "live_search"
    case geminiFlash = "gemini_flash"
    case geminiSearch = "gemini_search"
    case auto

    static var allCases: [LLMMode] {
        [.legacyFast, .liveSearch, .geminiSearch, .auto]
    }

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .legacyFast:
            return "Llama 3.1 8B Instant (Cutoff)"
        case .liveSearch:
            return "GPT-4o Mini Search Preview (Live Search)"
        case .geminiFlash:
            return "Gemini 2.5 Flash Lite (Cutoff)"
        case .geminiSearch:
            return "Gemini 2.5 Flash Lite (Live Search)"
        case .auto:
            return "Auto (Mixed)"
        }
    }
}

struct GenerateRequest: Codable, Hashable {
    let text: String
    var llmMode: LLMMode?
    var llmModel: String?
    var speaker: String?
    var profileType: ProfileType?
    var avatarProfile: String?
    var lipsyncBackend: LipsyncBackend?
    var modelPath: String?
    var refWavPath: String?
    var maxChunkChars: Int?
    var maxChunkWords: Int?
    var avatarFPS: Double?
    var avatarMaxFrameEdge: Int?
    var musetalkBatchSize: Int?
    var musetalkInferFPS: Double?
    var musetalkStreamWindowSec: Double?
    var musetalkLookaheadSec: Double?
    var musetalkJPEGQuality: Int?
    var musetalkFaceScale: Double?
    var musetalkAudioHistorySec: Double?
    var musetalkMaxChunkChars: Int?
    var musetalkFirstChunkChars: Int?
    var pitchShift: Double?
    var paceScale: Double?
    var volumeGain: Double?
    var f0Scale: Double?
    var embeddingScale: Double?
    var deEsserCutoff: Double?
    var deEsserOrder: Int?

    enum CodingKeys: String, CodingKey {
        case text
        case llmMode = "llm_mode"
        case llmModel = "llm_model"
        case speaker
        case profileType = "profile_type"
        case avatarProfile = "avatar_profile"
        case lipsyncBackend = "lipsync_backend"
        case modelPath = "model_path"
        case refWavPath = "ref_wav_path"
        case maxChunkChars = "max_chunk_chars"
        case maxChunkWords = "max_chunk_words"
        case avatarFPS = "avatar_fps"
        case avatarMaxFrameEdge = "avatar_max_frame_edge"
        case musetalkBatchSize = "musetalk_batch_size"
        case musetalkInferFPS = "musetalk_infer_fps"
        case musetalkStreamWindowSec = "musetalk_stream_window_sec"
        case musetalkLookaheadSec = "musetalk_lookahead_sec"
        case musetalkJPEGQuality = "musetalk_jpeg_quality"
        case musetalkFaceScale = "musetalk_face_scale"
        case musetalkAudioHistorySec = "musetalk_audio_history_sec"
        case musetalkMaxChunkChars = "musetalk_max_chunk_chars"
        case musetalkFirstChunkChars = "musetalk_first_chunk_chars"
        case pitchShift = "pitch_shift"
        case paceScale = "pace_scale"
        case volumeGain = "volume_gain"
        case f0Scale = "f0_scale"
        case embeddingScale = "embedding_scale"
        case deEsserCutoff = "de_esser_cutoff"
        case deEsserOrder = "de_esser_order"
    }
}

struct NDJSONChunkEnvelope: Decodable {
    let event: String?
    let detail: String?
    let inferenceMS: Double?

    let chunkIndex: Int?
    let audioBase64: String?
    let sampleRate: Int?
    let fps: Double?
    let framesBase64: [String]?
    let durationSec: Double?

    enum CodingKeys: String, CodingKey {
        case event
        case detail
        case inferenceMS = "inference_ms"
        case chunkIndex = "chunk_index"
        case audioBase64 = "audio_base64"
        case sampleRate = "sample_rate"
        case fps
        case framesBase64 = "frames_base64"
        case durationSec = "duration_sec"
    }
}

struct DecodedStreamChunk {
    let chunkIndex: Int
    let audioBuffer: AVAudioPCMBuffer
    let sampleRate: Double
    let fps: Double?
    let framePayloads: [Data]
    let durationSec: Double
}

enum StreamingEvent {
    case chunk(DecodedStreamChunk)
    case done(inferenceMS: Double?)
}
