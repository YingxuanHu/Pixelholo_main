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
    @Published var transcript = ""
    @Published var errorMessage: String?

    private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private let audioEngine = AVAudioEngine()

    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?

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
        guard !isRecording else { return }
        errorMessage = nil
        transcript = ""

        stopTranscribing(commitResult: false)

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
            if let result {
                Task { @MainActor in
                    self.transcript = result.bestTranscription.formattedString
                }
                if result.isFinal {
                    Task { @MainActor in
                        self.stopTranscribing(commitResult: true)
                    }
                }
            }
            if let error {
                Task { @MainActor in
                    self.errorMessage = "Speech recognition error: \(error.localizedDescription)"
                    self.stopTranscribing(commitResult: false)
                }
            }
        }

        isRecording = true
    }

    @discardableResult
    func stopTranscribing(commitResult: Bool = true) -> String? {
        if audioEngine.isRunning {
            audioEngine.stop()
        }
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil

        isRecording = false
        deactivateAudioSessionIfPossible()
        let final = commitResult ? transcript.trimmingCharacters(in: .whitespacesAndNewlines) : nil
        return final?.isEmpty == false ? final : nil
    }

    private func configureAudioSession() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement, options: [.duckOthers, .defaultToSpeaker, .allowBluetoothHFP])
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
}
