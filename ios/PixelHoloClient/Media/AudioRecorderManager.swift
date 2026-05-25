import AVFoundation
import Combine
import Foundation

@MainActor
final class AudioRecorderManager: NSObject, ObservableObject {
    @Published private(set) var isRecording = false
    @Published private(set) var recordedURL: URL?
    @Published var errorMessage: String?

    private var recorder: AVAudioRecorder?

    func requestMicrophonePermission() async -> Bool {
        await withCheckedContinuation { continuation in
            AVAudioApplication.requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
    }

    func startRecording(profileName: String) async {
        errorMessage = nil
        do {
            try configureAudioSessionForRecording()
            let outputURL = Self.makeAudioURL(profileName: profileName)
            let settings: [String: Any] = [
                AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
                AVSampleRateKey: 44_100,
                AVNumberOfChannelsKey: 1,
                AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
            ]
            let recorder = try AVAudioRecorder(url: outputURL, settings: settings)
            recorder.delegate = self
            recorder.isMeteringEnabled = false
            guard recorder.record() else {
                throw NSError(domain: "AudioRecorder", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to start recording"])
            }
            self.recorder = recorder
            self.recordedURL = nil
            self.isRecording = true
        } catch {
            self.errorMessage = error.localizedDescription
            self.isRecording = false
        }
    }

    func stopRecording() {
        recorder?.stop()
        recorder = nil
        isRecording = false
        deactivateAudioSessionIfPossible()
    }

    private func configureAudioSessionForRecording() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetooth])
        try session.setActive(true)
    }

    private func deactivateAudioSessionIfPossible() {
        let session = AVAudioSession.sharedInstance()
        try? session.setActive(false, options: .notifyOthersOnDeactivation)
    }

    private static func makeAudioURL(profileName: String) -> URL {
        let safe = profileName
            .replacingOccurrences(of: " ", with: "_")
            .replacingOccurrences(of: "/", with: "_")
        let filename = "\(safe)_\(Int(Date().timeIntervalSince1970)).m4a"
        return FileManager.default.temporaryDirectory.appendingPathComponent(filename)
    }
}

extension AudioRecorderManager: AVAudioRecorderDelegate {
    nonisolated func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        Task { @MainActor in
            self.isRecording = false
            self.deactivateAudioSessionIfPossible()
            if flag {
                self.recordedURL = recorder.url
            } else {
                self.errorMessage = "Audio recording did not complete."
            }
        }
    }
}
