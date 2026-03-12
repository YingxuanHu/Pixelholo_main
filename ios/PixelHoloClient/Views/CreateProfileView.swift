import SwiftUI

@MainActor
struct CreateProfileView: View {
    private let contentInset: CGFloat = 14

    @EnvironmentObject private var appSession: AppSessionViewModel
    @EnvironmentObject private var serverConfig: ServerConfig

    @StateObject private var audioRecorder = AudioRecorderManager()
    @StateObject private var profilesViewModel = ProfileListViewModel()
    private let apiClient = APIClient()

    @State private var profileName = ""
    @State private var profileType: ProfileType = .avatar
    @State private var capturedVideoURL: URL?
    @State private var capturedAudioURL: URL?
    @State private var uploadedVideoFilename: String?
    @State private var uploadedAudioFilename: String?
    @State private var statusMessage: String?
    @State private var isUploadingVideo = false
    @State private var isUploadingAudio = false
    @State private var showsSetupSheet = false

    var body: some View {
        AppScreen {
            ScrollView(.vertical, showsIndicators: false) {
                VStack(spacing: 16) {
                    header

                    if let message = statusMessage {
                        AppBanner(
                            text: message,
                            tone: bannerTone(for: message)
                        )
                        .padding(.horizontal, contentInset)
                    }

                    workspaceSurface
                }
                .padding(.top, 8)
                .padding(.bottom, AppSpacing.bottomTabClearance)
            }
        }
        .toolbar(.hidden, for: .navigationBar)
        .sheet(isPresented: $showsSetupSheet) {
            setupSheet
        }
        .task(id: "\(serverConfig.baseURL?.absoluteString ?? "nil")|\(profileType.rawValue)") {
            await profilesViewModel.loadProfiles(baseURL: serverConfig.baseURL, reason: .automatic)
            applySessionDraft()
        }
        .onChange(of: audioRecorder.recordedURL) { _, value in
            if let value {
                capturedAudioURL = value
                statusMessage = "Audio captured: \(value.lastPathComponent)"
            }
        }
        .onChange(of: profileName) { _, _ in
            syncStagedProfile()
        }
        .onChange(of: profileType) { _, _ in
            syncStagedProfile()
            Task {
                await profilesViewModel.loadProfiles(baseURL: serverConfig.baseURL, reason: .manual)
            }
        }
    }

    private var header: some View {
        HStack(alignment: .top, spacing: 14) {
            AppScreenHeader(
                title: "Create",
                subtitle: "Capture training media on-device and upload it directly to the backend."
            )

            Spacer(minLength: 12)

            AppIconCircleButton(icon: "questionmark") {
                showsSetupSheet = true
            }
        }
        .padding(.horizontal, contentInset)
    }

    private var workspaceSurface: some View {
        AppPrimarySurface {
            identitySection

            AppSectionDivider()

            videoSection

            AppSectionDivider()

            audioSection

            continueSection
        }
        .padding(.horizontal, contentInset)
    }

    private var identitySection: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(trimmedProfileName.isEmpty ? "New profile" : trimmedProfileName)
                .font(.system(size: 28, weight: .bold, design: .rounded))
                .foregroundStyle(.primary)

            TextField("Name this profile first", text: $profileName)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled(true)
                .appInputField()

            Picker("Workflow", selection: $profileType) {
                Text("Voice").tag(ProfileType.voice)
                Text("Avatar").tag(ProfileType.avatar)
            }
            .pickerStyle(.segmented)

            HStack(spacing: 10) {
                AppMetricPill(
                    title: "Workflow",
                    value: profileType == .avatar ? "Avatar" : "Voice",
                    tint: profileType == .avatar ? .indigo : .blue
                )
                AppMetricPill(
                    title: "Video",
                    value: capturedVideoURL == nil ? "Missing" : "Ready",
                    tint: capturedVideoURL == nil ? .orange : .green
                )
                AppMetricPill(
                    title: "Audio",
                    value: capturedAudioURL == nil ? "Missing" : "Ready",
                    tint: capturedAudioURL == nil ? .orange : .green
                )
            }

            if !existingProfilesForCurrentType.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Use existing profile")
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(.secondary)

                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 8) {
                            ForEach(existingProfilesForCurrentType) { profile in
                                profileChip(profile)
                            }
                        }
                        .padding(.vertical, 2)
                    }
                }
            }

            Text(serverConfig.baseURL == nil
                 ? "Configure the backend URL before uploading."
                 : "Uploads will be saved on the backend under \(backendDestinationLabel).")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var videoSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Video Capture")
                .font(.title3.weight(.semibold))

            Text(profileType == .avatar
                 ? "Record a 10 second 3:4 clip for avatar workflows."
                 : "Video is optional for voice workflows, but you can still record and upload one.")
                .font(.footnote)
                .foregroundStyle(.secondary)

            if trimmedProfileName.isEmpty {
                AppBanner(
                    text: "Choose the profile name first. The captured file will then be uploaded into that backend profile.",
                    tone: .warning
                )
            } else if appSession.streamSession.isStreaming {
                AppBanner(
                    text: "Live streaming is active. Camera preview is paused so playback can continue. Stop the stream before recording video.",
                    tone: .neutral
                )
            } else if appSession.selectedTab == .create, profileType == .avatar {
                CameraView(profileName: trimmedProfileName.isEmpty ? "profile" : trimmedProfileName) { url in
                    capturedVideoURL = url
                    statusMessage = "Video captured: \(url.lastPathComponent)"
                }
            } else if profileType == .avatar {
                AppBanner(
                    text: "Open the Create tab and keep it active to start the live camera preview.",
                    tone: .neutral
                )
            } else {
                AppBanner(
                    text: "Video capture is optional for voice workflows.",
                    tone: .neutral
                )
            }

            if let capturedVideoURL {
                AppBanner(text: "Video ready: \(capturedVideoURL.lastPathComponent)", tone: .success)
            }

            if let uploadedVideoFilename {
                AppBanner(text: "Video uploaded to \(trimmedProfileName): \(uploadedVideoFilename)", tone: .success)
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
    }

    private var audioSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Audio Capture")
                .font(.title3.weight(.semibold))

            Text("Record a clean voice sample for training. 10 to 15 minutes is recommended for a useful profile.")
                .font(.footnote)
                .foregroundStyle(.secondary)

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
                    title: audioRecorder.isRecording ? "Stop & Save Audio" : "Record Audio",
                    icon: audioRecorder.isRecording ? "stop.circle.fill" : "mic.fill"
                )
            }
            .buttonStyle(.borderedProminent)
            .tint(audioRecorder.isRecording ? .red : .blue)
            .disabled(trimmedProfileName.isEmpty || appSession.streamSession.isStreaming)

            if trimmedProfileName.isEmpty {
                AppBanner(
                    text: "Choose the profile name first so the recording and upload are tied to the correct backend profile.",
                    tone: .warning
                )
            } else if appSession.streamSession.isStreaming {
                AppBanner(
                    text: "Audio capture is disabled while a stream is playing. Stop the stream before recording training audio.",
                    tone: .neutral
                )
            }

            if let recorded = audioRecorder.recordedURL {
                AppBanner(text: "Audio ready: \(recorded.lastPathComponent)", tone: .success)
                    .onAppear {
                        capturedAudioURL = recorded
                    }
            }

            if let uploadedAudioFilename {
                AppBanner(text: "Audio uploaded to \(trimmedProfileName): \(uploadedAudioFilename)", tone: .success)
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
    }

    private var continueSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            AppSectionDivider()

            Text("Next Step")
                .font(.title3.weight(.semibold))

            Text("Once the required media is uploaded, move straight into preprocess and training for this profile.")
                .font(.footnote)
                .foregroundStyle(.secondary)

            Button {
                appSession.stageProfileWorkflow(name: trimmedProfileName, type: profileType)
                appSession.selectedTab = .pipeline
            } label: {
                AppPrimaryActionLabel(title: "Continue to Train", icon: "arrow.right.circle.fill")
            }
            .buttonStyle(.borderedProminent)
            .disabled(!canContinueToTrain)
        }
    }

    private var setupSheet: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: AppSpacing.cardSpacing) {
                    AppCard {
                        AppSectionHeader(
                            title: "Capture Notes",
                            subtitle: "Workflow guidance for creating a usable profile on the phone."
                        )

                        Text("1. Name the profile first.\n2. Record the required media.\n3. Upload to the backend.\n4. Continue to Train.")
                            .font(.subheadline)

                        Text("Avatar profiles need both video and audio uploaded. Voice profiles need audio; video is optional.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)

                        Text("Audio should ideally be several minutes long. Short samples will work technically, but training quality will be limited.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)

                        if let baseURL = serverConfig.baseURL?.absoluteString {
                            Text("Uploads will be sent to \(baseURL)")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding(.horizontal, AppSpacing.screenPadding)
                .padding(.top, 16)
                .padding(.bottom, 32)
            }
            .navigationTitle("Setup")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        showsSetupSheet = false
                    }
                }
            }
        }
        .presentationDetents([.medium, .large])
    }

    private var trimmedProfileName: String {
        profileName.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var existingProfilesForCurrentType: [ProfileInfo] {
        switch profileType {
        case .voice:
            return profilesViewModel.voiceProfiles
        case .avatar:
            return profilesViewModel.avatarProfiles
        }
    }

    private var backendDestinationLabel: String {
        let root = profileType == .avatar ? "data/avatar_profiles" : "data/voice_profiles"
        return "\(root)/\(trimmedProfileName)"
    }

    private var canContinueToTrain: Bool {
        let hasUploadedAudio = uploadedAudioFilename != nil
        if profileType == .avatar {
            return hasUploadedAudio && uploadedVideoFilename != nil
        }
        return hasUploadedAudio
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
            uploadedVideoFilename = response.filename
            syncStagedProfile()
            await profilesViewModel.loadProfiles(baseURL: baseURL, reason: .manual)
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
            uploadedAudioFilename = response.filename
            syncStagedProfile()
            await profilesViewModel.loadProfiles(baseURL: baseURL, reason: .manual)
            statusMessage = "Audio uploaded: \(response.filename)"
        } catch {
            statusMessage = "Audio upload failed: \(error.localizedDescription)"
        }
    }

    private func syncStagedProfile() {
        guard !trimmedProfileName.isEmpty else { return }
        appSession.stageProfileWorkflow(name: trimmedProfileName, type: profileType)
    }

    private func applySessionDraft() {
        if let profile = appSession.selectedProfile {
            profileName = profile.name
            profileType = profile.profileType
            return
        }
        if let stagedName = appSession.stagedProfileName {
            profileName = stagedName
        }
        if let stagedType = appSession.stagedProfileType {
            profileType = stagedType
        }
    }

    @ViewBuilder
    private func profileChip(_ profile: ProfileInfo) -> some View {
        let isSelected = trimmedProfileName.caseInsensitiveCompare(profile.name) == .orderedSame
        Button {
            profileName = profile.name
            profileType = profile.profileType
            statusMessage = "Using existing profile: \(profile.name)"
        } label: {
            VStack(alignment: .leading, spacing: 2) {
                Text(profile.name)
                    .font(.footnote.weight(.semibold))
                Text(profile.hasProfile ? "Trained" : "Needs training")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(isSelected ? Color.accentColor.opacity(0.16) : Color(uiColor: .tertiarySystemBackground))
            )
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    NavigationStack {
        CreateProfileView()
            .environmentObject(ServerConfig.shared)
            .environmentObject(AppSessionViewModel())
    }
}
