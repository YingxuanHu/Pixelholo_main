@preconcurrency import AVFoundation
import Combine
@preconcurrency import CoreImage
@preconcurrency import Foundation

final class CameraManager: NSObject, ObservableObject, @unchecked Sendable {
    @Published private(set) var isAuthorized = false
    @Published private(set) var isConfigured = false
    @Published private(set) var isRunning = false
    @Published private(set) var isRecording = false
    @Published private(set) var remainingSeconds = 10
    @Published private(set) var recordedURL: URL?
    @Published var errorMessage: String?

    let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "pixelholo.camera.session", qos: .userInitiated)
    private let movieOutput = AVCaptureMovieFileOutput()

    private var countdownTimer: DispatchSourceTimer?
    private var recordingStartDate: Date?
    private var maxDuration: TimeInterval = 10

    func requestPermissions() async -> Bool {
        let video = await withCheckedContinuation { continuation in
            AVCaptureDevice.requestAccess(for: .video) { granted in
                continuation.resume(returning: granted)
            }
        }
        let audio = await withCheckedContinuation { continuation in
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                continuation.resume(returning: granted)
            }
        }
        let granted = video && audio
        isAuthorized = granted
        return granted
    }

    func configureIfNeeded() async {
        guard isAuthorized else {
            errorMessage = "Camera and microphone permission are required."
            return
        }
        guard !isConfigured else { return }

        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            sessionQueue.async { [weak self] in
                guard let self else {
                    done.resume()
                    return
                }

                do {
                    try self.configureSession()
                    Task { @MainActor [weak self] in
                        self?.isConfigured = true
                        done.resume()
                    }
                } catch {
                    Task { @MainActor [weak self] in
                        self?.errorMessage = error.localizedDescription
                        done.resume()
                    }
                }
            }
        }
    }

    func startSession() {
        guard isConfigured else { return }
        let session = self.session
        sessionQueue.async { [weak self] in
            guard let self else { return }
            guard !session.isRunning else { return }
            session.startRunning()
            Task { @MainActor [weak self] in
                self?.isRunning = true
            }
        }
    }

    func stopSession() {
        let session = self.session
        sessionQueue.async { [weak self] in
            guard let self else { return }
            guard session.isRunning else { return }
            session.stopRunning()
            Task { @MainActor [weak self] in
                self?.isRunning = false
            }
        }
    }

    func startRecording(profileName: String, duration: TimeInterval = 10) {
        guard isConfigured else { return }
        guard !movieOutput.isRecording else { return }
        errorMessage = nil
        recordedURL = nil
        maxDuration = duration
        remainingSeconds = Int(duration)

        let movieOutput = self.movieOutput
        sessionQueue.async { [weak self] in
            guard let self else { return }
            let outputURL = Self.makeRawVideoURL(profileName: profileName)
            movieOutput.maxRecordedDuration = CMTime(seconds: duration, preferredTimescale: 600)
            movieOutput.startRecording(to: outputURL, recordingDelegate: self)
        }
    }

    func stopRecording() {
        let movieOutput = self.movieOutput
        sessionQueue.async { [weak self] in
            guard self != nil else { return }
            movieOutput.stopRecording()
        }
    }

    private func configureSession() throws {
        session.beginConfiguration()
        session.sessionPreset = .high

        session.inputs.forEach { session.removeInput($0) }
        session.outputs.forEach { session.removeOutput($0) }

        guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front)
            ?? AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            session.commitConfiguration()
            throw NSError(domain: "CameraManager", code: -1, userInfo: [NSLocalizedDescriptionKey: "No camera device available."])
        }

        let videoInput = try AVCaptureDeviceInput(device: videoDevice)
        guard session.canAddInput(videoInput) else {
            session.commitConfiguration()
            throw NSError(domain: "CameraManager", code: -2, userInfo: [NSLocalizedDescriptionKey: "Unable to add camera input."])
        }
        session.addInput(videoInput)

        if let audioDevice = AVCaptureDevice.default(for: .audio) {
            let audioInput = try AVCaptureDeviceInput(device: audioDevice)
            if session.canAddInput(audioInput) {
                session.addInput(audioInput)
            }
        }

        guard session.canAddOutput(movieOutput) else {
            session.commitConfiguration()
            throw NSError(domain: "CameraManager", code: -3, userInfo: [NSLocalizedDescriptionKey: "Unable to add movie output."])
        }
        session.addOutput(movieOutput)
        movieOutput.movieFragmentInterval = .invalid

        if let connection = movieOutput.connection(with: .video) {
            if connection.isVideoMirroringSupported {
                connection.isVideoMirrored = (videoDevice.position == .front)
            }
            if connection.isVideoStabilizationSupported {
                connection.preferredVideoStabilizationMode = .auto
            }
        }

        session.commitConfiguration()
    }

    private func startCountdownTimer() {
        countdownTimer?.cancel()
        recordingStartDate = Date()
        let timer = DispatchSource.makeTimerSource(queue: .main)
        timer.schedule(deadline: .now(), repeating: .milliseconds(200))
        timer.setEventHandler { [weak self] in
                guard let self else { return }
                guard let start = self.recordingStartDate else { return }
                let elapsed = Date().timeIntervalSince(start)
            let left = max(0, Int(ceil(self.maxDuration - elapsed)))
            self.remainingSeconds = left
        }
        timer.resume()
        countdownTimer = timer
    }

    private func stopCountdownTimer() {
        countdownTimer?.cancel()
        countdownTimer = nil
        recordingStartDate = nil
    }

    private func handleRecordingFinished(tempURL: URL, error: Error?) {
        stopCountdownTimer()
        isRecording = false

        if let error {
            errorMessage = error.localizedDescription
            return
        }

        Task.detached(priority: .userInitiated) {
            do {
                let outputURL = try await Self.cropVideoToThreeByFour(inputURL: tempURL)
                try? FileManager.default.removeItem(at: tempURL)
                await MainActor.run {
                    self.recordedURL = outputURL
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = "Video crop failed: \(error.localizedDescription)"
                }
            }
        }
    }

    private static func makeRawVideoURL(profileName: String) -> URL {
        let safe = profileName
            .replacingOccurrences(of: " ", with: "_")
            .replacingOccurrences(of: "/", with: "_")
        let filename = "\(safe)_raw_\(Int(Date().timeIntervalSince1970)).mov"
        return FileManager.default.temporaryDirectory.appendingPathComponent(filename)
    }

    private static func makeCroppedVideoURL(inputURL: URL) -> URL {
        let filename = inputURL.deletingPathExtension().lastPathComponent + "_3x4.mp4"
        return FileManager.default.temporaryDirectory.appendingPathComponent(filename)
    }

    private static func cropVideoToThreeByFour(inputURL: URL) async throws -> URL {
        let asset = AVURLAsset(url: inputURL)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw NSError(domain: "CameraManager", code: -20, userInfo: [NSLocalizedDescriptionKey: "Video track missing."])
        }

        let naturalSize = try await track.load(.naturalSize)
        let preferredTransform = try await track.load(.preferredTransform)
        let duration = try await asset.load(.duration)
        let transformed = CGRect(origin: .zero, size: naturalSize).applying(preferredTransform)
        let orientedSize = CGSize(width: abs(transformed.width), height: abs(transformed.height))

        guard orientedSize.width > 0, orientedSize.height > 0 else {
            throw NSError(domain: "CameraManager", code: -21, userInfo: [NSLocalizedDescriptionKey: "Invalid video dimensions."])
        }

        let targetAspect = 3.0 / 4.0
        let targetWidth = min(orientedSize.width, orientedSize.height * targetAspect)
        let targetHeight = targetWidth / targetAspect
        let cropRect = CGRect(
            x: (orientedSize.width - targetWidth) / 2.0,
            y: (orientedSize.height - targetHeight) / 2.0,
            width: targetWidth,
            height: targetHeight
        )

        let filterComposition = AVMutableVideoComposition(asset: asset) { request in
            let source = request.sourceImage
            let sourceExtent = source.extent
            let x = sourceExtent.origin.x + cropRect.origin.x
            let y = sourceExtent.origin.y + cropRect.origin.y
            let centeredCrop = CGRect(x: x, y: y, width: cropRect.width, height: cropRect.height)
            let clamped = source.clampedToExtent()
            let cropped = clamped.cropped(to: centeredCrop)
            request.finish(with: cropped, context: nil)
        }
        filterComposition.renderSize = CGSize(width: cropRect.width, height: cropRect.height)
        let nominalFPS = try await track.load(.nominalFrameRate)
        if nominalFPS > 0 {
            filterComposition.frameDuration = CMTime(value: 1, timescale: CMTimeScale(nominalFPS))
        } else {
            filterComposition.frameDuration = CMTime(value: 1, timescale: 30)
        }

        let outputURL = makeCroppedVideoURL(inputURL: inputURL)
        try? FileManager.default.removeItem(at: outputURL)
        guard let exporter = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetHighestQuality) else {
            throw NSError(domain: "CameraManager", code: -22, userInfo: [NSLocalizedDescriptionKey: "Failed to initialize video exporter."])
        }
        exporter.outputURL = outputURL
        exporter.outputFileType = .mp4
        exporter.videoComposition = filterComposition
        exporter.timeRange = CMTimeRange(start: .zero, duration: duration)

        struct ExportBox: @unchecked Sendable {
            let exporter: AVAssetExportSession
        }
        let exportBox = ExportBox(exporter: exporter)

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            exportBox.exporter.exportAsynchronously {
                let exporter = exportBox.exporter
                switch exporter.status {
                case .completed:
                    continuation.resume()
                case .failed:
                    continuation.resume(throwing: exporter.error ?? NSError(domain: "CameraManager", code: -23, userInfo: [NSLocalizedDescriptionKey: "Video export failed."]))
                case .cancelled:
                    continuation.resume(throwing: NSError(domain: "CameraManager", code: -24, userInfo: [NSLocalizedDescriptionKey: "Video export cancelled."]))
                default:
                    continuation.resume(throwing: NSError(domain: "CameraManager", code: -25, userInfo: [NSLocalizedDescriptionKey: "Video export ended with status \(exporter.status.rawValue)."]))
                }
            }
        }

        return outputURL
    }
}

extension CameraManager: AVCaptureFileOutputRecordingDelegate {
    nonisolated func fileOutput(
        _ output: AVCaptureFileOutput,
        didStartRecordingTo fileURL: URL,
        from connections: [AVCaptureConnection]
    ) {
        DispatchQueue.main.async {
            self.isRecording = true
            self.startCountdownTimer()
        }
    }

    nonisolated func fileOutput(
        _ output: AVCaptureFileOutput,
        didFinishRecordingTo outputFileURL: URL,
        from connections: [AVCaptureConnection],
        error: Error?
    ) {
        DispatchQueue.main.async {
            self.handleRecordingFinished(tempURL: outputFileURL, error: error)
        }
    }
}
