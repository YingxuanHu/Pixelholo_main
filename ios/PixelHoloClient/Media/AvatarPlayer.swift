import AVFoundation
import QuartzCore
import UIKit

@MainActor
final class AvatarPlayer: ObservableObject {
    @Published private(set) var currentFrame: UIImage?
    @Published private(set) var isPlaying = false
    @Published var errorMessage: String?

    private let engine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()
    private var displayLink: CADisplayLink?

    private var frameTimeline: [UIImage] = []
    private var streamFPS: Double = 25
    private var connectedFormatSignature: String?

    init() {
        engine.attach(playerNode)
        engine.connect(playerNode, to: engine.mainMixerNode, format: nil)
        engine.prepare()
    }

    deinit {
        displayLink?.invalidate()
        playerNode.stop()
        if engine.isRunning {
            engine.stop()
        }
    }

    func enqueue(_ chunk: DecodedStreamChunk) {
        do {
            try ensureEngineReady(for: chunk.audioBuffer.format)
            if let fps = chunk.fps, fps > 0 {
                streamFPS = fps
            }
            if !chunk.frames.isEmpty {
                frameTimeline.append(contentsOf: chunk.frames)
            }

            playerNode.scheduleBuffer(chunk.audioBuffer, at: nil, options: [], completionHandler: nil)
            if !playerNode.isPlaying {
                playerNode.play()
                isPlaying = true
                startDisplayLinkIfNeeded()
            }
        } catch {
            errorMessage = "Playback error: \(error.localizedDescription)"
        }
    }

    func stop() {
        playerNode.stop()
        if engine.isRunning {
            engine.stop()
        }
        frameTimeline.removeAll()
        currentFrame = nil
        isPlaying = false
        errorMessage = nil
        stopDisplayLink()
    }

    private func ensureEngineReady(for format: AVAudioFormat) throws {
        let signature = "\(format.sampleRate)-\(format.channelCount)"
        if connectedFormatSignature != signature {
            playerNode.stop()
            if engine.isRunning {
                engine.stop()
            }
            engine.disconnectNodeOutput(playerNode)
            engine.connect(playerNode, to: engine.mainMixerNode, format: format)
            connectedFormatSignature = signature
        }
        if !engine.isRunning {
            try engine.start()
        }
    }

    private func startDisplayLinkIfNeeded() {
        guard displayLink == nil else { return }
        let link = CADisplayLink(target: self, selector: #selector(handleDisplayLink))
        link.add(to: .main, forMode: .common)
        displayLink = link
    }

    private func stopDisplayLink() {
        displayLink?.invalidate()
        displayLink = nil
    }

    @objc private func handleDisplayLink() {
        guard streamFPS > 0 else { return }
        guard let lastRenderTime = playerNode.lastRenderTime,
              let playerTime = playerNode.playerTime(forNodeTime: lastRenderTime) else {
            return
        }

        let elapsedSeconds = Double(playerTime.sampleTime) / playerTime.sampleRate
        let frameIndex = Int(elapsedSeconds * streamFPS)

        if frameIndex >= 0 && frameIndex < frameTimeline.count {
            currentFrame = frameTimeline[frameIndex]
        } else if frameIndex >= frameTimeline.count, let last = frameTimeline.last {
            currentFrame = last
        }

        if !playerNode.isPlaying, isPlaying {
            isPlaying = false
        }
    }
}
