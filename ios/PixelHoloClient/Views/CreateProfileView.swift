import SwiftUI

@MainActor
struct CreateProfileView: View {
    @EnvironmentObject private var serverConfig: ServerConfig

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
        AppScreen {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: AppSpacing.cardSpacing) {
                    AppScreenHeader(
                        title: "Create",
                        subtitle: "Capture training media on-device and upload it directly to the backend."
                    )

                    AppCard {
                        AppSectionHeader(
                            title: "Step 1: Define the Profile",
                            subtitle: "Choose the workflow type first so the upload targets match the backend profile structure."
                        )

                        TextField("Profile name", text: $profileName)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled(true)
                            .appInputField()

                        VStack(alignment: .leading, spacing: 8) {
                            Text("Workflow")
                                .font(.subheadline.weight(.medium))
                                .foregroundStyle(.secondary)

                            Picker("Workflow", selection: $profileType) {
                                Text("Voice").tag(ProfileType.voice)
                                Text("Avatar").tag(ProfileType.avatar)
                            }
                            .pickerStyle(.segmented)
                        }

                        if let baseURL = serverConfig.baseURL?.absoluteString {
                            Text("Uploads will be sent to \(baseURL)")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                    }

                    AppCard {
                        AppSectionHeader(
                            title: "Step 2: Capture Video",
                            subtitle: "Record a 10 second 3:4 clip for avatar workflows. Voice-only profiles can skip this."
                        )

                        CameraView(profileName: trimmedProfileName.isEmpty ? "profile" : trimmedProfileName) { url in
                            capturedVideoURL = url
                            statusMessage = "Video captured: \(url.lastPathComponent)"
                        }

                        if let capturedVideoURL {
                            AppBanner(text: "Video ready: \(capturedVideoURL.lastPathComponent)", tone: .success)
                        }

                        Button {
                            Task {
                                await uploadVideo()
                            }
                        } label: {
                            AppPrimaryActionLabel(
                                title: isUploadingVideo ? "Uploading Video..." : "Upload Video",
                                icon: "video.fill.badge.plus"
                            )
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(isUploadingVideo || trimmedProfileName.isEmpty || capturedVideoURL == nil || serverConfig.baseURL == nil)
                    }

                    AppCard {
                        AppSectionHeader(
                            title: "Step 3: Capture Audio",
                            subtitle: "Record a clean voice sample for training. This is required for both voice and avatar profiles."
                        )

                        Button {
                            Task {
                                if audioRecorder.isRecording {
                                    audioRecorder.stopRecording()
                                } else {
                                    let granted = await audioRecorder.requestMicrophonePermission()
                                    if granted {
                                        await audioRecorder.startRecording(profileName: trimmedProfileName.isEmpty ? "profile" : trimmedProfileName)
                                    } else {
                                        statusMessage = "Microphone permission denied."
                                    }
                                }
                            }
                        } label: {
                            AppPrimaryActionLabel(
                                title: audioRecorder.isRecording ? "Stop Recording Audio" : "Record Audio",
                                icon: audioRecorder.isRecording ? "stop.circle.fill" : "mic.fill"
                            )
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(audioRecorder.isRecording ? .red : .blue)

                        if let recorded = audioRecorder.recordedURL {
                            AppBanner(text: "Audio ready: \(recorded.lastPathComponent)", tone: .success)
                                .onAppear {
                                    capturedAudioURL = recorded
                                }
                        }

                        Button {
                            Task {
                                await uploadAudio()
                            }
                        } label: {
                            AppPrimaryActionLabel(
                                title: isUploadingAudio ? "Uploading Audio..." : "Upload Audio",
                                icon: "waveform.badge.plus"
                            )
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(isUploadingAudio || trimmedProfileName.isEmpty || capturedAudioURL == nil || serverConfig.baseURL == nil)
                    }

                    if let message = statusMessage {
                        AppBanner(
                            text: message,
                            tone: bannerTone(for: message)
                        )
                    }
                }
                .padding(.horizontal, AppSpacing.screenPadding)
                .padding(.top, 12)
                .padding(.bottom, AppSpacing.bottomTabClearance)
            }
        }
        .toolbar(.hidden, for: .navigationBar)
        .onChange(of: audioRecorder.recordedURL) { _, value in
            if let value {
                capturedAudioURL = value
                statusMessage = "Audio captured: \(value.lastPathComponent)"
            }
        }
    }

    private var trimmedProfileName: String {
        profileName.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func bannerTone(for message: String) -> AppBannerTone {
        let lowercased = message.lowercased()
        if lowercased.contains("failed") || lowercased.contains("denied") || lowercased.contains("invalid") {
            return .error
        }
        if lowercased.contains("uploaded") || lowercased.contains("captured") {
            return .success
        }
        return .neutral
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
        guard !trimmedProfileName.isEmpty else {
            statusMessage = "Profile name is required."
            return
        }

        isUploadingVideo = true
        defer { isUploadingVideo = false }

        do {
            let response = try await apiClient.uploadVideo(
                baseURL: baseURL,
                fileURL: capturedVideoURL,
                profile: trimmedProfileName,
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
        guard !trimmedProfileName.isEmpty else {
            statusMessage = "Profile name is required."
            return
        }

        isUploadingAudio = true
        defer { isUploadingAudio = false }

        do {
            let response = try await apiClient.uploadAudio(
                baseURL: baseURL,
                fileURL: capturedAudioURL,
                profile: trimmedProfileName,
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
