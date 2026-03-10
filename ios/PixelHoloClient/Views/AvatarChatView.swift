import SwiftUI

struct AvatarChatView: View {
    @EnvironmentObject private var appSession: AppSessionViewModel
    @EnvironmentObject private var serverConfig: ServerConfig
    @StateObject private var viewModel: AvatarChatViewModel
    @StateObject private var speech = SpeechRecognizerManager()
    @State private var inputText = ""
    @State private var speechPressIsActive = false

    init(initialProfile: ProfileInfo? = nil) {
        _viewModel = StateObject(wrappedValue: AvatarChatViewModel(initialProfile: initialProfile))
    }

    var body: some View {
        AppScreen {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: AppSpacing.cardSpacing) {
                    AppScreenHeader(
                        title: "Stream",
                        subtitle: "Send text or speech to the backend and render the live output here."
                    )

                    previewCard
                    sessionCard
                    promptCard

                    if let error = viewModel.errorMessage ?? viewModel.player.errorMessage {
                        AppBanner(text: error, tone: .error)
                    }

                    if let speechError = speech.errorMessage {
                        AppBanner(text: speechError, tone: .error)
                    }

                    AppCard {
                        ConsoleLogView(logs: viewModel.logs, title: "Stream Logs")
                    }
                }
                .padding(.horizontal, AppSpacing.screenPadding)
                .padding(.top, 12)
                .padding(.bottom, AppSpacing.bottomTabClearance)
            }
        }
        .toolbar(.hidden, for: .navigationBar)
        .onDisappear {
            viewModel.stopStreaming()
            speechPressIsActive = false
            _ = speech.stopTranscribing(commitResult: false)
        }
        .task {
            _ = await speech.requestPermissions()
            viewModel.applySelectedProfile(appSession.selectedProfile)
        }
        .onChange(of: appSession.selectedProfile?.id) { _, _ in
            viewModel.applySelectedProfile(appSession.selectedProfile)
        }
    }

    private var previewCard: some View {
        AppCard {
            AppSectionHeader(
                title: "Live Output",
                subtitle: viewModel.profileType == .avatar
                    ? "Audio and avatar frames will render here in sync."
                    : "Audio-only mode is selected. The player is ready for speech output."
            )

            ZStack {
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .fill(Color.black)

                if let frame = viewModel.player.currentFrame {
                    AvatarFrameView(image: frame, cornerRadius: 20)
                } else {
                    VStack(spacing: 10) {
                        Image(systemName: viewModel.profileType == .avatar ? "person.crop.rectangle" : "speaker.wave.3.fill")
                            .font(.title2)
                            .foregroundStyle(.white.opacity(0.82))

                        Text(viewModel.isStreaming ? "Waiting for first chunk..." : "Ready")
                            .font(.subheadline.weight(.medium))
                            .foregroundStyle(.white.opacity(0.82))
                    }
                }
            }
            .aspectRatio(viewModel.profileType == .avatar ? 3.0 / 4.0 : 16.0 / 9.0, contentMode: .fit)

            HStack(spacing: 10) {
                AppMetricPill(title: "Mode", value: viewModel.profileType == .avatar ? "Avatar" : "Voice", tint: .blue)
                AppMetricPill(title: "Endpoint", value: viewModel.endpoint.rawValue.capitalized, tint: .indigo)
                AppMetricPill(title: "State", value: viewModel.isStreaming ? "Live" : "Idle", tint: viewModel.isStreaming ? .green : .gray)
            }
        }
    }

    private var sessionCard: some View {
        AppCard {
            AppSectionHeader(
                title: "Session Setup",
                subtitle: "Choose the target profile and output mode before sending a prompt."
            )

            VStack(alignment: .leading, spacing: 10) {
                Text("Profile")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.secondary)

                TextField("Enter a profile name or tap one from Profiles", text: $viewModel.profileName)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled(true)
                    .appInputField()
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Output")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.secondary)

                Picker("Output", selection: $viewModel.profileType) {
                    Text("Voice").tag(ProfileType.voice)
                    Text("Avatar").tag(ProfileType.avatar)
                }
                .pickerStyle(.segmented)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Endpoint")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.secondary)

                Picker("Endpoint", selection: $viewModel.endpoint) {
                    Text("Chat").tag(StreamEndpoint.chat)
                    Text("Speak").tag(StreamEndpoint.speak)
                }
                .pickerStyle(.segmented)
            }

            if viewModel.profileType == .avatar {
                AppKeyValueRow("Lip-Sync Backend") {
                    Picker("Lip-Sync Backend", selection: $viewModel.lipsyncBackend) {
                        ForEach(LipsyncBackend.allCases, id: \.self) { backend in
                            Text(backend.rawValue.capitalized).tag(backend)
                        }
                    }
                    .pickerStyle(.menu)
                }
            }

            if let baseURL = serverConfig.baseURL?.absoluteString {
                Text("Server: \(baseURL)")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var promptCard: some View {
        AppCard {
            AppSectionHeader(
                title: "Prompt",
                subtitle: "Type text manually or hold the microphone button to capture speech."
            )

            AppTextEditorBox(
                text: $inputText,
                placeholder: "Type something like: Introduce yourself in one sentence.",
                minHeight: 150
            )
            .textInputAutocapitalization(.sentences)

            HStack(spacing: 12) {
                Button {
                    viewModel.startStreaming(baseURL: serverConfig.baseURL, text: inputText)
                } label: {
                    AppPrimaryActionLabel(
                        title: viewModel.isStreaming ? "Streaming..." : "Send",
                        icon: "paperplane.fill"
                    )
                }
                .buttonStyle(.borderedProminent)
                .disabled(viewModel.isStreaming)

                Button {
                    Task {
                        await viewModel.interrupt(baseURL: serverConfig.baseURL)
                    }
                } label: {
                    AppPrimaryActionLabel(
                        title: "Stop",
                        icon: "stop.fill"
                    )
                }
                .buttonStyle(.bordered)
            }

            Button {
                // gesture-driven button; action intentionally empty
            } label: {
                AppPrimaryActionLabel(
                    title: speech.isRecording ? "Listening... release to send" : "Hold to Talk",
                    icon: speech.isRecording ? "waveform.circle.fill" : "mic.fill"
                )
            }
            .buttonStyle(.borderedProminent)
            .tint(speech.isRecording ? .red : .blue)
            .simultaneousGesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in
                        guard !speechPressIsActive else { return }
                        speechPressIsActive = true
                        Task { await startSpeechIfNeeded() }
                    }
                    .onEnded { _ in
                        speechPressIsActive = false
                        finishSpeechAndSend()
                    }
            )

            Text("For a quick sanity check, choose a trained profile and use `Speak` first before trying chat.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    private func startSpeechIfNeeded() async {
        guard !speech.isRecording, !speech.isStarting else { return }
        if !speech.isAuthorized {
            _ = await speech.requestPermissions()
        }
        await speech.startTranscribing()
    }

    private func finishSpeechAndSend() {
        guard speech.isRecording else { return }
        if let result = speech.stopTranscribing(commitResult: true) {
            inputText = result
            viewModel.endpoint = .chat
            viewModel.startStreaming(baseURL: serverConfig.baseURL, text: result)
        }
    }
}

#Preview {
    NavigationStack {
        AvatarChatView()
            .environmentObject(ServerConfig.shared)
            .environmentObject(AppSessionViewModel())
    }
}
