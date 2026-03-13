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

struct GenerateRequest: Codable, Hashable {
    let text: String
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
    var f0Scale: Double?
    var embeddingScale: Double?
    var diffusionSteps: Int?
    var deEsserCutoff: Double?
    var deEsserOrder: Int?

    enum CodingKeys: String, CodingKey {
        case text
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
        case f0Scale = "f0_scale"
        case embeddingScale = "embedding_scale"
        case diffusionSteps = "diffusion_steps"
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
