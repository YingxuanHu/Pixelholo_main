import AVFoundation
import Combine
import Foundation
import Speech

private enum SpeechRecognitionFailure: LocalizedError {
    case recognizerUnavailable
    case invalidAudioInput(inputSampleRate: Double, inputChannels: AVAudioChannelCount, outputSampleRate: Double, outputChannels: AVAudioChannelCount)

    var errorDescription: String? {
        switch self {
        case .recognizerUnavailable:
            return "Speech recognition is currently unavailable."
        case let .invalidAudioInput(inputSampleRate, inputChannels, outputSampleRate, outputChannels):
#if targetEnvironment(simulator)
            return "Push-to-talk could not start because the iOS Simulator is not exposing a valid microphone input format. Use typed prompts here, or test speech input on a physical iPhone."
#else
            return "Push-to-talk could not start because no valid microphone input was available. Input format: \(Int(inputSampleRate)) Hz / \(inputChannels) ch. Output format: \(Int(outputSampleRate)) Hz / \(outputChannels) ch."
#endif
        }
    }
}

@MainActor
final class SpeechRecognizerManager: NSObject, ObservableObject {
    @Published private(set) var isAuthorized = false
    @Published private(set) var isRecording = false
    @Published private(set) var isStarting = false
    @Published var transcript = ""
    @Published var errorMessage: String?

    private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private let audioEngine = AVAudioEngine()

    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private var recognitionSessionID: UInt64 = 0

    func requestPermissions() async -> Bool {
        let speechAuth = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }
        let micAllowed = await withCheckedContinuation { continuation in
            AVAudioApplication.requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
        isAuthorized = (speechAuth == .authorized) && micAllowed
        return isAuthorized
    }

    func startTranscribing() async {
        guard isAuthorized else {
            errorMessage = "Speech permission not granted."
            return
        }
        guard let recognizer, recognizer.isAvailable else {
            errorMessage = SpeechRecognitionFailure.recognizerUnavailable.localizedDescription
            return
        }
        guard !isRecording, !isStarting else { return }
        isStarting = true
        errorMessage = nil
        transcript = ""
        recognitionSessionID &+= 1
        let sessionID = recognitionSessionID
        _ = stopTranscribingInternal(commitResult: false)

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.requiresOnDeviceRecognition = false
        recognitionRequest = request

        do {
            try configureAudioSession()
            let inputNode = audioEngine.inputNode
            let recordingFormat = try resolveRecordingFormat(for: inputNode)
            inputNode.removeTap(onBus: 0)
            inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
                self?.recognitionRequest?.append(buffer)
            }

            audioEngine.prepare()
            try audioEngine.start()
        } catch {
            errorMessage = "Speech engine failed: \(error.localizedDescription)"
            stopTranscribing(commitResult: false)
            return
        }

        recognitionTask = recognizer.recognitionTask(with: request) { [weak self] result, error in
            guard let self else { return }
            guard self.recognitionSessionID == sessionID else { return }
            if let result {
                Task { @MainActor in
                    self.transcript = result.bestTranscription.formattedString
                }
                if result.isFinal {
                    Task { @MainActor in
                        self.completeSession(sessionID: sessionID, commitResult: true)
                    }
                }
            }
            if let error {
                Task { @MainActor in
                    guard self.recognitionSessionID == sessionID else { return }
                    if Self.shouldIgnoreSpeechRecognitionError(error, transcript: self.transcript) {
                        return
                    }
                    self.errorMessage = Self.describeSpeechRecognitionError(error)
                    self.completeSession(sessionID: sessionID, commitResult: false)
                }
            }
        }

        isRecording = true
        isStarting = false
    }

    @discardableResult
    func stopTranscribing(commitResult: Bool = true) -> String? {
        recognitionSessionID &+= 1
        return stopTranscribingInternal(commitResult: commitResult)
    }

    @discardableResult
    private func stopTranscribingInternal(commitResult: Bool) -> String? {
        let hadActiveRecognition =
            isRecording ||
            isStarting ||
            recognitionTask != nil ||
            recognitionRequest != nil ||
            audioEngine.isRunning

        guard hadActiveRecognition else {
            isRecording = false
            isStarting = false
            return nil
        }

        if audioEngine.isRunning {
            audioEngine.stop()
        }
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil

        isRecording = false
        isStarting = false
        deactivateAudioSessionIfPossible()
        let final = commitResult ? transcript.trimmingCharacters(in: .whitespacesAndNewlines) : nil
        return final?.isEmpty == false ? final : nil
    }

    @discardableResult
    private func completeSession(sessionID: UInt64, commitResult: Bool) -> String? {
        guard recognitionSessionID == sessionID else { return nil }
        recognitionSessionID &+= 1
        if commitResult {
            errorMessage = nil
        }
        return stopTranscribingInternal(commitResult: commitResult)
    }

    private func configureAudioSession() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement, options: [.duckOthers, .defaultToSpeaker, .allowBluetooth])
        try session.setActive(true, options: .notifyOthersOnDeactivation)
    }

    private func deactivateAudioSessionIfPossible() {
        let session = AVAudioSession.sharedInstance()
        try? session.setActive(false, options: .notifyOthersOnDeactivation)
    }

    private func resolveRecordingFormat(for inputNode: AVAudioInputNode) throws -> AVAudioFormat {
        let inputFormat = inputNode.inputFormat(forBus: 0)
        if isValidRecordingFormat(inputFormat) {
            return inputFormat
        }

        let outputFormat = inputNode.outputFormat(forBus: 0)
        if isValidRecordingFormat(outputFormat) {
            return outputFormat
        }

        throw SpeechRecognitionFailure.invalidAudioInput(
            inputSampleRate: inputFormat.sampleRate,
            inputChannels: inputFormat.channelCount,
            outputSampleRate: outputFormat.sampleRate,
            outputChannels: outputFormat.channelCount
        )
    }

    private func isValidRecordingFormat(_ format: AVAudioFormat) -> Bool {
        format.sampleRate > 0 && format.channelCount > 0
    }

    private static func describeSpeechRecognitionError(_ error: Error) -> String {
        let nsError = error as NSError
        if nsError.domain == "kAFAssistantErrorDomain", nsError.code == 216 {
            return "Speech recognition error: Apple speech recognition could not start cleanly. This usually happens when recording was triggered more than once for the same press, or the speech service was not ready yet. Release and try again."
        }
        if nsError.domain == "kAFAssistantErrorDomain", nsError.code == 203 {
            return "Speech recognition error: Apple speech recognition asked to retry. This is usually a transient Siri speech service issue. Release and try again."
        }
        return "Speech recognition error: \(error.localizedDescription)"
    }

    private static func shouldIgnoreSpeechRecognitionError(_ error: Error, transcript: String) -> Bool {
        let nsError = error as NSError
        if nsError.domain == NSURLErrorDomain, nsError.code == NSURLErrorCancelled {
            return true
        }
        if nsError.domain == "kAFAssistantErrorDomain", nsError.code == 216 {
            return !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        if nsError.domain == "kAFAssistantErrorDomain", nsError.code == 203 {
            return !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        return false
    }
}
