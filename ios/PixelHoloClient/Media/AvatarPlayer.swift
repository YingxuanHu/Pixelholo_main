import AVFoundation
import Combine
import ImageIO
import QuartzCore
import UIKit

@MainActor
final class AvatarPlayer: ObservableObject {
    @Published private(set) var currentFrame: UIImage?
    @Published private(set) var isPlaying = false
    @Published var errorMessage: String?

    private final class FrameSegment {
        let startTime: Double
        let duration: Double
        let framePayloads: [Data]
        private var decodedFrames: [UIImage?]
        private let lock = NSLock()

        init(startTime: Double, duration: Double, framePayloads: [Data]) {
            self.startTime = startTime
            self.duration = duration
            self.framePayloads = framePayloads
            self.decodedFrames = Array(repeating: nil, count: framePayloads.count)
        }

        var endTime: Double { startTime + duration }
        var frameCount: Int { framePayloads.count }

        func setDecodedFrame(_ image: UIImage?, at index: Int) {
            lock.lock()
            if decodedFrames.indices.contains(index) {
                decodedFrames[index] = image
            }
            lock.unlock()
        }

        func decodedFrame(at index: Int) -> UIImage? {
            lock.lock()
            defer { lock.unlock() }
            guard decodedFrames.indices.contains(index) else { return nil }
            return decodedFrames[index]
        }

        func lastDecodedFrame(upTo index: Int) -> UIImage? {
            lock.lock()
            defer { lock.unlock() }
            guard !decodedFrames.isEmpty else { return nil }
            let capped = min(index, decodedFrames.count - 1)
            guard capped >= 0 else { return nil }
            for i in stride(from: capped, through: 0, by: -1) {
                if let image = decodedFrames[i] {
                    return image
                }
            }
            return nil
        }

        func lastDecodedFrame() -> UIImage? {
            lock.lock()
            defer { lock.unlock() }
            for image in decodedFrames.reversed() {
                if let image {
                    return image
                }
            }
            return nil
        }

        func decodedPrefixCount() -> Int {
            lock.lock()
            defer { lock.unlock() }
            var prefix = 0
            for image in decodedFrames {
                guard image != nil else { break }
                prefix += 1
            }
            return prefix
        }

        func decodedCoverageDuration() -> Double {
            guard duration > 0, frameCount > 0 else { return 0 }
            return duration * (Double(decodedPrefixCount()) / Double(frameCount))
        }
    }

    private let engine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()
    private var displayLink: CADisplayLink?
    private var connectedFormatSignature: String?
    private let frameDecodeQueue: OperationQueue = {
        let queue = OperationQueue()
        queue.name = "pixelholo.avatar-frame-decode"
        queue.qualityOfService = .userInitiated
        queue.maxConcurrentOperationCount = max(2, min(4, ProcessInfo.processInfo.activeProcessorCount - 1))
        return queue
    }()

    private var frameSegments: [FrameSegment] = []
    private var scheduledAudioDuration: Double = 0
    private var hasAvatarPayload = false
    private var streamCompleted = false
    private var lastPresentedSegmentTime: (start: Double, index: Int)?
    private var pausedForFrameBuffering = false

    private let avatarPrerollSec = 0.45
    private let voicePrerollSec = 0.08
    private let minDecodedLeadToStartSec = 0.18
    private let minDecodedLeadToResumeSec = 0.16
    private let minDecodedLeadWhilePlayingSec = 0.05
    private static let maxDecodedFrameDimension: CGFloat = 384

    init() {
        engine.attach(playerNode)
        engine.connect(playerNode, to: engine.mainMixerNode, format: nil)
        engine.prepare()
    }

    deinit {
        displayLink?.invalidate()
        frameDecodeQueue.cancelAllOperations()
        playerNode.stop()
        if engine.isRunning {
            engine.stop()
        }
    }

    func enqueue(_ chunk: DecodedStreamChunk) {
        do {
            try ensureEngineReady(for: chunk.audioBuffer.format)

            let duration = max(
                chunk.durationSec,
                Double(chunk.audioBuffer.frameLength) / chunk.audioBuffer.format.sampleRate
            )
            let startTime = scheduledAudioDuration
            scheduledAudioDuration += duration

            if chunk.fps != nil || !chunk.framePayloads.isEmpty {
                hasAvatarPayload = true
            }
            if !chunk.framePayloads.isEmpty {
                let segment = FrameSegment(
                    startTime: startTime,
                    duration: duration,
                    framePayloads: chunk.framePayloads
                )
                frameSegments.append(segment)
                scheduleFrameDecoding(for: segment)
            }

            playerNode.scheduleBuffer(chunk.audioBuffer, at: nil, options: [], completionHandler: nil)
            startDisplayLinkIfNeeded()
            startPlaybackIfReady()
        } catch {
            errorMessage = "Playback error: \(error.localizedDescription)"
        }
    }

    private func scheduleFrameDecoding(for segment: FrameSegment) {
        let maxPixelSize = Self.maxDecodedFrameDimension
        for (index, payload) in segment.framePayloads.enumerated() {
            let operation = BlockOperation { [weak self, weak segment] in
                guard let self, let segment else { return }
                autoreleasepool {
                    let image = Self.decodeFrameImage(from: payload, maxPixelSize: maxPixelSize)
                    segment.setDecodedFrame(image, at: index)
                    DispatchQueue.main.async { [weak self, weak segment] in
                        guard let self, let segment else { return }
                        if self.currentFrame == nil, let first = segment.decodedFrame(at: index) {
                            self.currentFrame = first
                        }
                        self.startPlaybackIfReady()
                    }
                }
            }
            operation.queuePriority = index < 2 ? .veryHigh : .normal
            frameDecodeQueue.addOperation(operation)
        }
    }

    func finishStream() {
        streamCompleted = true
        startPlaybackIfReady()
    }

    func stop() {
        playerNode.stop()
        if engine.isRunning {
            engine.stop()
        }
        frameDecodeQueue.cancelAllOperations()
        frameSegments.removeAll()
        scheduledAudioDuration = 0
        hasAvatarPayload = false
        streamCompleted = false
        lastPresentedSegmentTime = nil
        pausedForFrameBuffering = false
        currentFrame = nil
        isPlaying = false
        errorMessage = nil
        stopDisplayLink()
        deactivatePlaybackSessionIfPossible()
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
            try configurePlaybackSession()
            try engine.start()
        }
    }

    private func configurePlaybackSession() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .moviePlayback, options: [.duckOthers])
        try session.setActive(true, options: .notifyOthersOnDeactivation)
    }

    private func deactivatePlaybackSessionIfPossible() {
        let session = AVAudioSession.sharedInstance()
        try? session.setActive(false, options: .notifyOthersOnDeactivation)
    }

    private func startPlaybackIfReady() {
        let playbackTime = currentPlaybackTime()
        let decodedLead = decodedCoverageAhead(of: playbackTime)

        if pausedForFrameBuffering {
            let canResume = streamCompleted ? decodedLead > 0.01 : decodedLead >= minDecodedLeadToResumeSec
            guard canResume else { return }
            playerNode.play()
            pausedForFrameBuffering = false
            isPlaying = true
            return
        }

        guard !playerNode.isPlaying else {
            isPlaying = true
            return
        }

        let bufferedDuration = scheduledAudioDuration - playbackTime
        let requiredPreroll = hasAvatarPayload ? avatarPrerollSec : voicePrerollSec

        let readyToStart: Bool
        if streamCompleted {
            readyToStart = bufferedDuration > 0.01 && (!hasAvatarPayload || decodedLead > 0.01)
        } else if hasAvatarPayload {
            readyToStart = bufferedDuration >= requiredPreroll
                && !frameSegments.isEmpty
                && decodedLead >= minDecodedLeadToStartSec
        } else {
            readyToStart = bufferedDuration >= requiredPreroll
        }

        guard readyToStart else { return }

        playerNode.play()
        isPlaying = true
    }

    private func currentPlaybackTime() -> Double {
        guard let lastRenderTime = playerNode.lastRenderTime,
              let playerTime = playerNode.playerTime(forNodeTime: lastRenderTime) else {
            return 0
        }
        return Double(playerTime.sampleTime) / playerTime.sampleRate
    }

    private func startDisplayLinkIfNeeded() {
        guard displayLink == nil else { return }
        let link = CADisplayLink(target: self, selector: #selector(handleDisplayLink))
        if #available(iOS 15.0, *) {
            link.preferredFrameRateRange = CAFrameRateRange(minimum: 30, maximum: 60, preferred: 60)
        } else {
            link.preferredFramesPerSecond = 60
        }
        link.add(to: .main, forMode: .common)
        displayLink = link
    }

    private func stopDisplayLink() {
        displayLink?.invalidate()
        displayLink = nil
    }

    @objc private func handleDisplayLink() {
        let playbackTime = currentPlaybackTime()
        managePlaybackBuffering(playbackTime: playbackTime)
        updateCurrentFrame(for: playbackTime)
        trimSegmentsIfNeeded(playbackTime: playbackTime)

        if !playerNode.isPlaying, isPlaying, !pausedForFrameBuffering {
            isPlaying = false
            if streamCompleted && playbackTime >= scheduledAudioDuration - 0.02 {
                stopDisplayLink()
            }
        }
    }

    private func managePlaybackBuffering(playbackTime: Double) {
        guard hasAvatarPayload else { return }

        let decodedLead = decodedCoverageAhead(of: playbackTime)

        if playerNode.isPlaying, !streamCompleted, decodedLead < minDecodedLeadWhilePlayingSec {
            playerNode.pause()
            pausedForFrameBuffering = true
            isPlaying = false
            return
        }

        if pausedForFrameBuffering {
            let canResume = streamCompleted ? decodedLead > 0.01 : decodedLead >= minDecodedLeadToResumeSec
            if canResume {
                playerNode.play()
                pausedForFrameBuffering = false
                isPlaying = true
            }
        }
    }

    private func updateCurrentFrame(for playbackTime: Double) {
        guard hasAvatarPayload else { return }
        guard !frameSegments.isEmpty else { return }

        if let segment = frameSegments.first(where: { playbackTime >= $0.startTime && playbackTime < $0.endTime }) {
            let localTime = max(0, min(playbackTime - segment.startTime, segment.duration))
            let framesPerSecond = segment.duration > 0
                ? Double(segment.frameCount) / segment.duration
                : Double(segment.frameCount)
            let frameIndex = max(0, min(segment.frameCount - 1, Int(floor(localTime * max(framesPerSecond, 1)))))
            let presentationKey = (segment.startTime, frameIndex)
            if lastPresentedSegmentTime?.start != presentationKey.0 || lastPresentedSegmentTime?.index != presentationKey.1 {
                if let image = segment.decodedFrame(at: frameIndex) ?? segment.lastDecodedFrame(upTo: frameIndex) {
                    currentFrame = image
                }
                lastPresentedSegmentTime = presentationKey
            }
            return
        }

        if playbackTime >= (frameSegments.last?.endTime ?? 0), let lastFrame = frameSegments.last?.lastDecodedFrame() {
            currentFrame = lastFrame
        }
    }

    private func trimSegmentsIfNeeded(playbackTime: Double) {
        while frameSegments.count > 2,
              let first = frameSegments.first,
              first.endTime < playbackTime - 0.25 {
            frameSegments.removeFirst()
        }
    }

    private func decodedCoverageAhead(of playbackTime: Double) -> Double {
        guard hasAvatarPayload else { return .infinity }
        guard !frameSegments.isEmpty else { return 0 }

        var coverage = 0.0
        var foundStart = false

        for segment in frameSegments {
            let decodedDuration = segment.decodedCoverageDuration()
            if !foundStart {
                guard playbackTime < segment.endTime else { continue }
                foundStart = true
                let localTime = max(0, playbackTime - segment.startTime)
                coverage += max(0, decodedDuration - localTime)
                if decodedDuration + 0.001 < segment.duration {
                    break
                }
            } else {
                coverage += decodedDuration
                if decodedDuration + 0.001 < segment.duration {
                    break
                }
            }
        }

        return foundStart ? coverage : 0
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
