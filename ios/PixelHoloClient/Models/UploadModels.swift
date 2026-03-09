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
