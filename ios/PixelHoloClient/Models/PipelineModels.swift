import Foundation

struct PreprocessRequest: Codable, Hashable {
    let profile: String
    var filename: String?
    var audioFilename: String?
    var profileType: ProfileType
    var bakeAvatar: Bool
    var avatarFPS: Double?
    var avatarStartSec: Double?
    var avatarLoopSec: Double?
    var avatarLoopFadeSec: Double?
    var avatarResizeFactor: Int?
    var avatarPads: String?
    var avatarBatchSize: Int?
    var avatarNosmooth: Bool
    var avatarBlurBackground: Bool?
    var avatarBlurKernel: Int?
    var avatarDevice: String?

    enum CodingKeys: String, CodingKey {
        case profile
        case filename
        case audioFilename = "audio_filename"
        case profileType = "profile_type"
        case bakeAvatar = "bake_avatar"
        case avatarFPS = "avatar_fps"
        case avatarStartSec = "avatar_start_sec"
        case avatarLoopSec = "avatar_loop_sec"
        case avatarLoopFadeSec = "avatar_loop_fade_sec"
        case avatarResizeFactor = "avatar_resize_factor"
        case avatarPads = "avatar_pads"
        case avatarBatchSize = "avatar_batch_size"
        case avatarNosmooth = "avatar_nosmooth"
        case avatarBlurBackground = "avatar_blur_background"
        case avatarBlurKernel = "avatar_blur_kernel"
        case avatarDevice = "avatar_device"
    }
}

struct TrainRequest: Codable, Hashable {
    let profile: String
    var profileType: ProfileType
    var batchSize: Int?
    var epochs: Int?
    var maxLen: Int?
    var autoSelectEpoch: Bool = true
    var autoTuneProfile: Bool = true
    var autoBuildLexicon: Bool = true
    var selectThorough: Bool = true
    var earlyStop: Bool = true

    enum CodingKeys: String, CodingKey {
        case profile
        case profileType = "profile_type"
        case batchSize = "batch_size"
        case epochs
        case maxLen = "max_len"
        case autoSelectEpoch = "auto_select_epoch"
        case autoTuneProfile = "auto_tune_profile"
        case autoBuildLexicon = "auto_build_lexicon"
        case selectThorough = "select_thorough"
        case earlyStop = "early_stop"
    }
}

struct ConsoleLogLine: Identifiable, Hashable {
    let id = UUID()
    let timestamp: Date
    let text: String
    let isError: Bool
}

