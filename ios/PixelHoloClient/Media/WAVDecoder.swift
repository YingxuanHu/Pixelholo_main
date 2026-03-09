import AVFoundation
import Foundation

enum WAVDecoderError: LocalizedError {
    case invalidHeader
    case missingFormatChunk
    case missingDataChunk
    case unsupportedFormat(audioFormat: UInt16, bitsPerSample: UInt16)
    case invalidData

    var errorDescription: String? {
        switch self {
        case .invalidHeader:
            return "Invalid WAV header."
        case .missingFormatChunk:
            return "WAV fmt chunk is missing."
        case .missingDataChunk:
            return "WAV data chunk is missing."
        case let .unsupportedFormat(audioFormat, bitsPerSample):
            return "Unsupported WAV format audioFormat=\(audioFormat), bitsPerSample=\(bitsPerSample)."
        case .invalidData:
            return "Invalid WAV payload."
        }
    }
}

enum WAVDecoder {
    struct FormatInfo {
        let audioFormat: UInt16
        let channels: UInt16
        let sampleRate: UInt32
        let bitsPerSample: UInt16
    }

    static func decodePCMBuffer(from wavData: Data) throws -> AVAudioPCMBuffer {
        guard wavData.count >= 44 else {
            throw WAVDecoderError.invalidHeader
        }
        guard String(data: wavData.subdata(in: 0..<4), encoding: .ascii) == "RIFF",
              String(data: wavData.subdata(in: 8..<12), encoding: .ascii) == "WAVE" else {
            throw WAVDecoderError.invalidHeader
        }

        var offset = 12
        var formatInfo: FormatInfo?
        var audioPayload: Data?

        while offset + 8 <= wavData.count {
            let idRange = offset..<(offset + 4)
            let sizeRange = (offset + 4)..<(offset + 8)
            guard let chunkID = String(data: wavData.subdata(in: idRange), encoding: .ascii) else {
                break
            }
            let chunkSize = Int(readUInt32LE(wavData, at: sizeRange.lowerBound))
            let payloadStart = offset + 8
            let payloadEnd = payloadStart + chunkSize
            guard payloadEnd <= wavData.count else {
                throw WAVDecoderError.invalidData
            }

            let payload = wavData.subdata(in: payloadStart..<payloadEnd)
            if chunkID == "fmt " {
                if payload.count < 16 { throw WAVDecoderError.invalidData }
                formatInfo = FormatInfo(
                    audioFormat: readUInt16LE(payload, at: 0),
                    channels: readUInt16LE(payload, at: 2),
                    sampleRate: readUInt32LE(payload, at: 4),
                    bitsPerSample: readUInt16LE(payload, at: 14)
                )
            } else if chunkID == "data" {
                audioPayload = payload
            }

            let paddedChunkSize = chunkSize + (chunkSize % 2)
            offset = payloadStart + paddedChunkSize
        }

        guard let formatInfo else { throw WAVDecoderError.missingFormatChunk }
        guard let audioPayload else { throw WAVDecoderError.missingDataChunk }

        guard formatInfo.audioFormat == 1, formatInfo.bitsPerSample == 16 else {
            throw WAVDecoderError.unsupportedFormat(
                audioFormat: formatInfo.audioFormat,
                bitsPerSample: formatInfo.bitsPerSample
            )
        }

        let channelCount = Int(formatInfo.channels)
        guard channelCount > 0 else { throw WAVDecoderError.invalidData }

        let totalSamples = audioPayload.count / MemoryLayout<Int16>.size
        guard totalSamples > 0 else { throw WAVDecoderError.invalidData }
        guard totalSamples % channelCount == 0 else { throw WAVDecoderError.invalidData }

        let frameCount = totalSamples / channelCount
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(formatInfo.sampleRate),
            channels: AVAudioChannelCount(channelCount),
            interleaved: false
        ) else {
            throw WAVDecoderError.invalidData
        }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frameCount)) else {
            throw WAVDecoderError.invalidData
        }
        buffer.frameLength = AVAudioFrameCount(frameCount)

        let samples = audioPayload.withUnsafeBytes { rawBuffer -> [Int16] in
            let ptr = rawBuffer.bindMemory(to: Int16.self)
            return Array(ptr)
        }
        guard let channels = buffer.floatChannelData else {
            throw WAVDecoderError.invalidData
        }

        for frame in 0..<frameCount {
            let base = frame * channelCount
            for channel in 0..<channelCount {
                let sample = samples[base + channel]
                channels[channel][frame] = max(-1.0, min(1.0, Float(sample) / Float(Int16.max)))
            }
        }

        return buffer
    }

    private static func readUInt16LE(_ data: Data, at offset: Int) -> UInt16 {
        let b0 = UInt16(data[offset])
        let b1 = UInt16(data[offset + 1]) << 8
        return b0 | b1
    }

    private static func readUInt32LE(_ data: Data, at offset: Int) -> UInt32 {
        let b0 = UInt32(data[offset])
        let b1 = UInt32(data[offset + 1]) << 8
        let b2 = UInt32(data[offset + 2]) << 16
        let b3 = UInt32(data[offset + 3]) << 24
        return b0 | b1 | b2 | b3
    }
}

