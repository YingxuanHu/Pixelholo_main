import Foundation

enum ProfileType: String, Codable, CaseIterable {
    case voice
    case avatar

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let raw = (try? container.decode(String.self))?.lowercased() ?? ""
        self = ProfileType(rawValue: raw) ?? .voice
    }
}

struct ProfileInfo: Codable, Identifiable, Hashable {
    let name: String
    let profileType: ProfileType
    let hasData: Bool
    let rawFiles: Int
    let rawAudioFiles: Int
    let processedWavs: Int
    let hasProfile: Bool
    let bestCheckpoint: String?
    let latestCheckpoint: String?

    var id: String {
        "\(profileType.rawValue):\(name)"
    }

    enum CodingKeys: String, CodingKey {
        case name
        case profileType = "profile_type"
        case hasData = "has_data"
        case rawFiles = "raw_files"
        case rawAudioFiles = "raw_audio_files"
        case processedWavs = "processed_wavs"
        case hasProfile = "has_profile"
        case bestCheckpoint = "best_checkpoint"
        case latestCheckpoint = "latest_checkpoint"
    }

    init(
        name: String,
        profileType: ProfileType,
        hasData: Bool,
        rawFiles: Int,
        rawAudioFiles: Int,
        processedWavs: Int,
        hasProfile: Bool,
        bestCheckpoint: String?,
        latestCheckpoint: String?
    ) {
        self.name = name
        self.profileType = profileType
        self.hasData = hasData
        self.rawFiles = rawFiles
        self.rawAudioFiles = rawAudioFiles
        self.processedWavs = processedWavs
        self.hasProfile = hasProfile
        self.bestCheckpoint = bestCheckpoint
        self.latestCheckpoint = latestCheckpoint
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.name = try container.decode(String.self, forKey: .name)
        self.profileType = try container.decodeIfPresent(ProfileType.self, forKey: .profileType) ?? .voice
        self.hasData = try container.decodeIfPresent(Bool.self, forKey: .hasData) ?? false
        self.rawFiles = try container.decodeIfPresent(Int.self, forKey: .rawFiles) ?? 0
        self.rawAudioFiles = try container.decodeIfPresent(Int.self, forKey: .rawAudioFiles) ?? 0
        self.processedWavs = try container.decodeIfPresent(Int.self, forKey: .processedWavs) ?? 0
        self.hasProfile = try container.decodeIfPresent(Bool.self, forKey: .hasProfile) ?? false
        self.bestCheckpoint = try container.decodeIfPresent(String.self, forKey: .bestCheckpoint)
        self.latestCheckpoint = try container.decodeIfPresent(String.self, forKey: .latestCheckpoint)
    }
}

struct ProfilesResponse: Codable {
    let profiles: [ProfileInfo]
}

