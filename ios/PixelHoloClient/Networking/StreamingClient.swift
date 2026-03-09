import AVFoundation
import Foundation
import UIKit

final class StreamingClient {
    private let lineStreamer: URLSessionLineStreamer
    private let decoder: JSONDecoder

    init(lineStreamer: URLSessionLineStreamer = URLSessionLineStreamer()) {
        self.lineStreamer = lineStreamer
        self.decoder = JSONDecoder()
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
        urlRequest.httpBody = try JSONEncoder().encode(request)

        let rawLines = lineStreamer.streamLines(request: urlRequest)

        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for try await line in rawLines {
                        try Task.checkCancellation()
                        let envelope = try decoder.decode(NDJSONChunkEnvelope.self, from: Data(line.utf8))

                        if envelope.event == "done" {
                            continuation.yield(.done(inferenceMS: envelope.inferenceMS))
                            continue
                        }
                        if envelope.event == "error" {
                            throw APIError.server(
                                statusCode: 500,
                                message: envelope.detail ?? "Streaming worker failed."
                            )
                        }

                        guard envelope.audioBase64 != nil else {
                            continue
                        }
                        let decodedChunk = try await decodeChunk(envelope)
                        continuation.yield(.chunk(decodedChunk))
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

    private func decodeChunk(_ envelope: NDJSONChunkEnvelope) async throws -> DecodedStreamChunk {
        try await Task.detached(priority: .utility) {
            guard let audioBase64 = envelope.audioBase64,
                  let audioData = Data(base64Encoded: audioBase64) else {
                throw APIError.decoding(WAVDecoderError.invalidData)
            }
            let pcmBuffer = try WAVDecoder.decodePCMBuffer(from: audioData)
            let sampleRate = envelope.sampleRate.map(Double.init) ?? pcmBuffer.format.sampleRate
            let duration = envelope.durationSec ?? (Double(pcmBuffer.frameLength) / pcmBuffer.format.sampleRate)

            let decodedFrames: [UIImage] = (envelope.framesBase64 ?? []).compactMap { encoded in
                guard let frameData = Data(base64Encoded: encoded) else { return nil }
                return UIImage(data: frameData)
            }

            return DecodedStreamChunk(
                chunkIndex: envelope.chunkIndex ?? 0,
                audioBuffer: pcmBuffer,
                sampleRate: sampleRate,
                fps: envelope.fps,
                frames: decodedFrames,
                durationSec: duration
            )
        }.value
    }
}

