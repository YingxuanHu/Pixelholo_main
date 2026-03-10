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
    private var lastPresentedAbsoluteFrameIndex = -1
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
                updateDisplayLinkRateIfNeeded()
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
        lastPresentedAbsoluteFrameIndex = -1
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
        applyPreferredRate(to: link)
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
            if absoluteFrameIndex != lastPresentedAbsoluteFrameIndex {
                currentFrame = frameTimeline[localFrameIndex]
                lastPresentedAbsoluteFrameIndex = absoluteFrameIndex
            }
        } else if localFrameIndex >= frameTimeline.count, let last = frameTimeline.last {
            if absoluteFrameIndex != lastPresentedAbsoluteFrameIndex {
                currentFrame = last
                lastPresentedAbsoluteFrameIndex = absoluteFrameIndex
            }
        }

        trimFramesIfNeeded(currentFrameIndex: absoluteFrameIndex)

        if !playerNode.isPlaying, isPlaying {
            isPlaying = false
        }
    }

    private func trimFramesIfNeeded(currentFrameIndex: Int) {
        guard frameTimeline.count > 120 else { return }
        let framesBehindCurrent = currentFrameIndex - trimmedFrameCount
        guard framesBehindCurrent > 60 else { return }

        let removableCount = min(framesBehindCurrent - 30, frameTimeline.count - 60)
        guard removableCount > 0 else { return }

        frameTimeline.removeFirst(removableCount)
        trimmedFrameCount += removableCount
    }

    private func updateDisplayLinkRateIfNeeded() {
        guard let displayLink else { return }
        applyPreferredRate(to: displayLink)
    }

    private func applyPreferredRate(to displayLink: CADisplayLink) {
        let clampedRate = max(15, min(60, Int(streamFPS.rounded())))
        if #available(iOS 15.0, *) {
            let rate = Float(clampedRate)
            displayLink.preferredFrameRateRange = CAFrameRateRange(
                minimum: rate,
                maximum: rate,
                preferred: rate
            )
        } else {
            displayLink.preferredFramesPerSecond = clampedRate
        }
    }
}
