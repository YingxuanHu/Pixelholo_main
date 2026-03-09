import AVFoundation
import Foundation
import Speech

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
            let recordingFormat = inputNode.outputFormat(forBus: 0)
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

        recognitionTask = recognizer?.recognitionTask(with: request) { [weak self] result, error in
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
        let final = commitResult ? transcript.trimmingCharacters(in: .whitespacesAndNewlines) : nil
        return final?.isEmpty == false ? final : nil
    }

    private func configureAudioSession() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement, options: [.duckOthers, .defaultToSpeaker, .allowBluetoothHFP])
        try session.setActive(true, options: .notifyOthersOnDeactivation)
    }
}

