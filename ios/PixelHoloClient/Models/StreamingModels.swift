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

    enum CodingKeys: String, CodingKey {
        case text
        case speaker
        case profileType = "profile_type"
        case avatarProfile = "avatar_profile"
        case lipsyncBackend = "lipsync_backend"
        case modelPath = "model_path"
        case refWavPath = "ref_wav_path"
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
    let frames: [UIImage]
    let durationSec: Double
}

enum StreamingEvent {
    case chunk(DecodedStreamChunk)
    case done(inferenceMS: Double?)
}

