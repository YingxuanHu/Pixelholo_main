import AVFoundation
import Combine
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
    private var trimmedFrameCount = 0
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
        trimmedFrameCount = 0
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
        let absoluteFrameIndex = Int(elapsedSeconds * streamFPS)
        let localFrameIndex = absoluteFrameIndex - trimmedFrameCount

        if localFrameIndex >= 0 && localFrameIndex < frameTimeline.count {
            currentFrame = frameTimeline[localFrameIndex]
        } else if localFrameIndex >= frameTimeline.count, let last = frameTimeline.last {
            currentFrame = last
        }

        trimFramesIfNeeded(currentFrameIndex: absoluteFrameIndex)

        if !playerNode.isPlaying, isPlaying {
            isPlaying = false
        }
    }

    private func trimFramesIfNeeded(currentFrameIndex: Int) {
        guard frameTimeline.count > 180 else { return }
        let framesBehindCurrent = currentFrameIndex - trimmedFrameCount
        guard framesBehindCurrent > 90 else { return }

        let removableCount = min(framesBehindCurrent - 45, frameTimeline.count - 90)
        guard removableCount > 0 else { return }

        frameTimeline.removeFirst(removableCount)
        trimmedFrameCount += removableCount
    }
}
