import AVFoundation
import Foundation
import ImageIO
import UIKit

private let pixelHoloBinaryStreamMagic = Data("PHS1".utf8)
private let pixelHoloBinaryStreamHeaderLength = 12
private let pixelHoloBinaryStreamMediaType = "application/vnd.pixelholo.stream-v1"

private struct BinaryStreamPacketMetadata: Decodable {
    let event: String?
    let detail: String?
    let inferenceMS: Double?
    let chunkIndex: Int?
    let sampleRate: Int?
    let fps: Double?
    let durationSec: Double?
    let audioBytesLen: Int?
    let frameLengths: [Int]?

    enum CodingKeys: String, CodingKey {
        case event
        case detail
        case inferenceMS = "inference_ms"
        case chunkIndex = "chunk_index"
        case sampleRate = "sample_rate"
        case fps
        case durationSec = "duration_sec"
        case audioBytesLen = "audio_bytes_len"
        case frameLengths = "frame_lengths"
    }
}

private struct BinaryStreamPacket {
    let metadata: BinaryStreamPacketMetadata
    let payload: Data
}

private struct BinaryPacketParser {
    private var buffer = Data()
    private let decoder = JSONDecoder()

    mutating func append(_ data: Data) {
        buffer.append(data)
    }

    mutating func nextPacket() throws -> BinaryStreamPacket? {
        guard buffer.count >= pixelHoloBinaryStreamHeaderLength else {
            return nil
        }
        guard buffer.prefix(4) == pixelHoloBinaryStreamMagic else {
            throw APIError.invalidResponse
        }

        let metadataLength = Int(readUInt32(at: 4))
        let payloadLength = Int(readUInt32(at: 8))
        let packetLength = pixelHoloBinaryStreamHeaderLength + metadataLength + payloadLength
        guard buffer.count >= packetLength else {
            return nil
        }

        let metadataStart = pixelHoloBinaryStreamHeaderLength
        let metadataEnd = metadataStart + metadataLength
        let payloadStart = metadataEnd
        let payloadEnd = payloadStart + payloadLength

        let metadataData = buffer.subdata(in: metadataStart..<metadataEnd)
        let payload = buffer.subdata(in: payloadStart..<payloadEnd)
        buffer.removeSubrange(0..<payloadEnd)

        do {
            let metadata = try decoder.decode(BinaryStreamPacketMetadata.self, from: metadataData)
            return BinaryStreamPacket(metadata: metadata, payload: payload)
        } catch {
            throw APIError.decoding(error)
        }
    }

    private func readUInt32(at offset: Int) -> UInt32 {
        let slice = buffer[offset..<(offset + 4)]
        return slice.reduce(0) { partial, byte in
            (partial << 8) | UInt32(byte)
        }
    }
}

final class StreamingClient {
    private let binaryStreamer: URLSessionBinaryStreamer
    private static let maxDecodedFrameDimension: CGFloat = 960

    init(binaryStreamer: URLSessionBinaryStreamer = URLSessionBinaryStreamer()) {
        self.binaryStreamer = binaryStreamer
    }

    func stream(
        baseURL: URL,
        endpoint: StreamEndpoint,
        request: GenerateRequest
    ) throws -> AsyncThrowingStream<StreamingEvent, Error> {
        let endpointURL = baseURL.appendingPathComponent(endpoint.rawValue)
        var urlRequest = URLRequest(url: endpointURL)
        urlRequest.httpMethod = "POST"
        urlRequest.timeoutInterval = 60 * 60
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.setValue(pixelHoloBinaryStreamMediaType, forHTTPHeaderField: "Accept")
        urlRequest.setValue("binary", forHTTPHeaderField: "X-PixelHolo-Transport")
        urlRequest.httpBody = try JSONEncoder().encode(request)

        let rawChunks = binaryStreamer.streamData(request: urlRequest)

        return AsyncThrowingStream { continuation in
            let task = Task {
                var parser = BinaryPacketParser()
                do {
                    for try await data in rawChunks {
                        try Task.checkCancellation()
                        parser.append(data)
                        while let packet = try parser.nextPacket() {
                            if packet.metadata.event == "done" {
                                continuation.yield(.done(inferenceMS: packet.metadata.inferenceMS))
                                continue
                            }
                            if packet.metadata.event == "error" {
                                throw APIError.server(
                                    statusCode: 500,
                                    message: packet.metadata.detail ?? "Streaming worker failed."
                                )
                            }
                            let decodedChunk = try await decodeChunk(packet)
                            continuation.yield(.chunk(decodedChunk))
                        }
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    private func decodeChunk(_ packet: BinaryStreamPacket) async throws -> DecodedStreamChunk {
        let maxDecodedFrameDimension = Self.maxDecodedFrameDimension
        return try await Task.detached(priority: .utility) {
            let metadata = packet.metadata
            let audioLength = metadata.audioBytesLen ?? 0
            guard audioLength >= 0, packet.payload.count >= audioLength else {
                throw APIError.invalidResponse
            }

            let audioData = packet.payload.subdata(in: 0..<audioLength)
            let framePayload = packet.payload.subdata(in: audioLength..<packet.payload.count)
            let pcmBuffer = try WAVDecoder.decodePCMBuffer(from: audioData)
            let sampleRate = metadata.sampleRate.map(Double.init) ?? pcmBuffer.format.sampleRate
            let duration = metadata.durationSec ?? (Double(pcmBuffer.frameLength) / pcmBuffer.format.sampleRate)

            var frames: [UIImage] = []
            var cursor = 0
            for frameLength in metadata.frameLengths ?? [] {
                guard frameLength >= 0, cursor + frameLength <= framePayload.count else {
                    throw APIError.invalidResponse
                }
                let frameData = framePayload.subdata(in: cursor..<(cursor + frameLength))
                cursor += frameLength
                if let image = Self.decodeFrameImage(from: frameData, maxPixelSize: maxDecodedFrameDimension) {
                    frames.append(image)
                }
            }

            return DecodedStreamChunk(
                chunkIndex: metadata.chunkIndex ?? 0,
                audioBuffer: pcmBuffer,
                sampleRate: sampleRate,
                fps: metadata.fps,
                frames: frames,
                durationSec: duration
            )
        }.value
    }

    nonisolated private static func decodeFrameImage(from data: Data, maxPixelSize: CGFloat) -> UIImage? {
        let options: [CFString: Any] = [
            kCGImageSourceShouldCache: false,
        ]
        guard let source = CGImageSourceCreateWithData(data as CFData, options as CFDictionary) else {
            return UIImage(data: data)
        }

        let thumbnailOptions: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceShouldCacheImmediately: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixelSize,
        ]
        if let cgImage = CGImageSourceCreateThumbnailAtIndex(
            source,
            0,
            thumbnailOptions as CFDictionary
        ) {
            return UIImage(cgImage: cgImage)
        }
        return UIImage(data: data)
    }
}

