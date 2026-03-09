import SwiftUI

@MainActor
struct CreateProfileView: View {
    @EnvironmentObject private var serverConfig: ServerConfig
    @Environment(\.dismiss) private var dismiss

    @StateObject private var audioRecorder = AudioRecorderManager()
    private let apiClient = APIClient()

    @State private var profileName = ""
    @State private var profileType: ProfileType = .avatar
    @State private var capturedVideoURL: URL?
    @State private var capturedAudioURL: URL?
    @State private var statusMessage: String?
    @State private var isUploadingVideo = false
    @State private var isUploadingAudio = false

    var body: some View {
        Form {
            Section("Profile") {
                TextField("Profile name", text: $profileName)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled(true)

                Picker("Workflow", selection: $profileType) {
                    Text("Voice").tag(ProfileType.voice)
                    Text("Avatar").tag(ProfileType.avatar)
                }
                .pickerStyle(.segmented)
            }

            Section("Video Capture (3:4, 10s)") {
                CameraView(profileName: profileName.isEmpty ? "profile" : profileName) { url in
                    capturedVideoURL = url
                    statusMessage = "Video captured: \(url.lastPathComponent)"
                }
                if let capturedVideoURL {
                    Text("Captured: \(capturedVideoURL.lastPathComponent)")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                Button {
                    Task {
                        await uploadVideo()
                    }
                } label: {
                    Text(isUploadingVideo ? "Uploading video..." : "Upload Video")
                        .frame(maxWidth: .infinity)
                }
                .disabled(isUploadingVideo || profileName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || capturedVideoURL == nil || serverConfig.baseURL == nil)
            }

            Section("Audio Capture") {
                Button {
                    Task {
                        if audioRecorder.isRecording {
                            audioRecorder.stopRecording()
                        } else {
                            let granted = await audioRecorder.requestMicrophonePermission()
                            if granted {
                                await audioRecorder.startRecording(profileName: profileName.isEmpty ? "profile" : profileName)
                            } else {
                                statusMessage = "Microphone permission denied."
                            }
                        }
                    }
                } label: {
                    Text(audioRecorder.isRecording ? "Stop Recording Audio" : "Record Audio")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)

                if let recorded = audioRecorder.recordedURL {
                    Text("Captured: \(recorded.lastPathComponent)")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .onAppear {
                            capturedAudioURL = recorded
                        }
                }

                Button {
                    Task {
                        await uploadAudio()
                    }
                } label: {
                    Text(isUploadingAudio ? "Uploading audio..." : "Upload Audio")
                        .frame(maxWidth: .infinity)
                }
                .disabled(isUploadingAudio || profileName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || capturedAudioURL == nil || serverConfig.baseURL == nil)
            }

            if let message = statusMessage {
                Section("Status") {
                    Text(message)
                        .foregroundStyle(message.lowercased().contains("failed") ? .red : .secondary)
                }
            }
        }
        .navigationTitle("Create Profile")
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button("Done") {
                    dismiss()
                }
            }
        }
        .onChange(of: audioRecorder.recordedURL) { _, value in
            if let value {
                capturedAudioURL = value
                statusMessage = "Audio captured: \(value.lastPathComponent)"
            }
        }
    }

    private func uploadVideo() async {
        guard !isUploadingVideo else { return }
        guard let baseURL = serverConfig.baseURL else {
            statusMessage = "Invalid server URL."
            return
        }
        guard let capturedVideoURL else {
            statusMessage = "No video captured yet."
            return
        }
        let name = profileName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else {
            statusMessage = "Profile name is required."
            return
        }

        isUploadingVideo = true
        defer { isUploadingVideo = false }

        do {
            let response = try await apiClient.uploadVideo(
                baseURL: baseURL,
                fileURL: capturedVideoURL,
                profile: name,
                profileType: profileType
            )
            statusMessage = "Video uploaded: \(response.filename)"
        } catch {
            statusMessage = "Video upload failed: \(error.localizedDescription)"
        }
    }

    private func uploadAudio() async {
        guard !isUploadingAudio else { return }
        guard let baseURL = serverConfig.baseURL else {
            statusMessage = "Invalid server URL."
            return
        }
        guard let capturedAudioURL else {
            statusMessage = "No audio captured yet."
            return
        }
        let name = profileName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else {
            statusMessage = "Profile name is required."
            return
        }

        isUploadingAudio = true
        defer { isUploadingAudio = false }

        do {
            let response = try await apiClient.uploadAudio(
                baseURL: baseURL,
                fileURL: capturedAudioURL,
                profile: name,
                profileType: profileType
            )
            statusMessage = "Audio uploaded: \(response.filename)"
        } catch {
            statusMessage = "Audio upload failed: \(error.localizedDescription)"
        }
    }
}

#Preview {
    NavigationStack {
        CreateProfileView()
            .environmentObject(ServerConfig.shared)
    }
}

