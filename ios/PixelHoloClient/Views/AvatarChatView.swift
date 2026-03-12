import SwiftUI

struct AvatarChatView: View {
    private let contentHorizontalInset: CGFloat = 14
    private let previewCornerRadius: CGFloat = 18

    @EnvironmentObject private var appSession: AppSessionViewModel
    @EnvironmentObject private var serverConfig: ServerConfig
    @EnvironmentObject private var viewModel: AvatarChatViewModel
    @StateObject private var speech = SpeechRecognizerManager()
    @State private var inputText = ""
    @State private var speechPressIsActive = false
    @State private var showsSessionOptions = false
    @State private var showsDiagnostics = false
    @FocusState private var inputFocused: Bool

    var body: some View {
        AppScreen {
            GeometryReader { proxy in
                let contentWidth = max(0, proxy.size.width - (contentHorizontalInset * 2))

                ScrollView(.vertical, showsIndicators: false) {
                    VStack(spacing: 12) {
                        stageHeader
                        previewStage(contentWidth: contentWidth)
                        composerDock(contentWidth: contentWidth)
                    }
                    .frame(maxWidth: .infinity, minHeight: proxy.size.height, alignment: .top)
                    .contentShape(Rectangle())
                    .padding(.bottom, AppSpacing.bottomTabClearance)
                }
                .scrollDismissesKeyboard(.interactively)
                .onTapGesture {
                    dismissKeyboard()
                }
            }
        }
        .toolbar(.hidden, for: .navigationBar)
        .sheet(isPresented: $showsSessionOptions) {
            sessionOptionsSheet
        }
        .sheet(isPresented: $showsDiagnostics) {
            diagnosticsSheet
        }
        .onDisappear {
            speechPressIsActive = false
            if speech.isRecording || speech.isStarting {
                _ = speech.stopTranscribing(commitResult: false)
            }
        }
        .task {
            _ = await speech.requestPermissions()
            viewModel.applySelectedProfile(appSession.selectedProfile)
            viewModel.prepareSelectedProfileWarmup(baseURL: serverConfig.baseURL)
        }
        .onChange(of: appSession.selectedProfile?.id) { _, _ in
            viewModel.applySelectedProfile(appSession.selectedProfile)
            viewModel.prepareSelectedProfileWarmup(baseURL: serverConfig.baseURL)
        }
        .onChange(of: speech.transcript) { _, newValue in
            guard speech.isRecording || speech.isStarting else { return }
            inputText = newValue
        }
    }

    private var stageHeader: some View {
        HStack(alignment: .top, spacing: 14) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Stream")
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                    .foregroundStyle(.primary)
                Text(viewModel.profileName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "Pick a profile and talk." : viewModel.profileName)
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 12)

            HStack(spacing: 10) {
                AppIconCircleButton(icon: "slider.horizontal.3") {
                    dismissKeyboard()
                    showsSessionOptions = true
                }
                AppIconCircleButton(icon: "text.alignleft") {
                    dismissKeyboard()
                    showsDiagnostics = true
                }
            }
        }
        .padding(.horizontal, contentHorizontalInset)
        .padding(.top, 4)
        .padding(.bottom, 6)
    }

    private func previewStage(contentWidth: CGFloat) -> some View {
        let stageWidth = viewModel.profileType == .avatar ? contentWidth : contentWidth
        let stageHeight = viewModel.profileType == .avatar
            ? stageWidth * (4.0 / 3.0)
            : stageWidth * (9.0 / 16.0)

        return ZStack {
            RoundedRectangle(cornerRadius: previewCornerRadius, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [
                            Color(uiColor: .secondarySystemGroupedBackground),
                            Color(uiColor: .systemGroupedBackground),
                        ],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )

            AvatarFrameView(player: viewModel.player, cornerRadius: previewCornerRadius)

            previewStatusOverlay
        }
        .frame(width: stageWidth, height: stageHeight)
        .frame(maxWidth: .infinity, alignment: .center)
        .clipShape(RoundedRectangle(cornerRadius: previewCornerRadius, style: .continuous))
        .overlay(alignment: .topLeading) {
            if let banner = currentBanner {
                AppBanner(text: banner.text, tone: banner.tone)
                    .padding(.horizontal, 10)
                    .padding(.top, 10)
            }
        }
    }

    @ViewBuilder
    private var previewStatusOverlay: some View {
        if viewModel.player.currentFrame != nil {
            EmptyView()
        } else if viewModel.isWarmingUp && !viewModel.isStreaming {
            ZStack {
                Rectangle()
                    .fill(Color.black.opacity(0.18))

                VStack(spacing: 12) {
                    ProgressView()
                        .progressViewStyle(.circular)
                        .tint(.white)
                        .scaleEffect(1.15)

                    Text("Preparing \(viewModel.profileName.isEmpty ? "profile" : viewModel.profileName)")
                        .font(.headline.weight(.semibold))
                        .foregroundStyle(.white)

                    Text("Warming the backend so the first response starts at normal speed.")
                        .font(.footnote)
                        .multilineTextAlignment(.center)
                        .foregroundStyle(.white.opacity(0.9))
                        .padding(.horizontal, 28)
                }
                .padding(.horizontal, 24)
            }
        } else {
            VStack {
                Spacer()

                VStack(alignment: .leading, spacing: 8) {
                    Text(stageEmptyTitle)
                        .font(.headline.weight(.semibold))
                        .foregroundStyle(.primary)

                    Text(stageEmptySubtitle)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .frame(maxWidth: 320, alignment: .leading)
                .padding(.horizontal, 18)
                .padding(.vertical, 16)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 20, style: .continuous)
                        .stroke(Color.white.opacity(0.18), lineWidth: 1)
                )
                .padding(.horizontal, 20)
                .padding(.bottom, 20)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
        }
    }

    private var stageEmptyTitle: String {
        if viewModel.isStreaming {
            return "Generating the first chunk"
        }
        if viewModel.profileName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "Pick a profile to begin"
        }
        return "Ready to stream"
    }

    private var stageEmptySubtitle: String {
        if viewModel.isStreaming {
            return "Audio and avatar frames will appear here as soon as the stream starts."
        }
        if viewModel.profileName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "Select a profile from Profiles or enter one in Session Options."
        }
        return "Send a prompt or hold to talk. The avatar preview will appear here."
    }

    private func composerDock(contentWidth: CGFloat) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            TextField(
                "",
                text: $inputText,
                prompt: Text(speech.isRecording ? "Listening..." : "Type something like: Introduce yourself in one sentence.")
                    .foregroundStyle(.secondary),
                axis: .vertical
            )
            .focused($inputFocused)
            .lineLimit(2)
            .textInputAutocapitalization(.sentences)
            .autocorrectionDisabled(false)
            .font(.body)
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
            .frame(minHeight: 72, alignment: .topLeading)
            .background(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(Color(uiColor: .tertiarySystemBackground))
            )

            holdToTalkControl

            HStack(spacing: 12) {
                Button {
                    dismissKeyboard()
                    viewModel.startStreaming(baseURL: serverConfig.baseURL, text: inputText)
                } label: {
                    AppPrimaryActionLabel(
                        title: viewModel.isWarmingUp && !viewModel.isStreaming ? "Preparing..." : (viewModel.isStreaming ? "Streaming..." : "Send"),
                        icon: "paperplane.fill"
                    )
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(viewModel.isStreaming || (viewModel.isWarmingUp && inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty))
                .frame(maxWidth: .infinity)

                Button {
                    dismissKeyboard()
                    Task {
                        await viewModel.interrupt(baseURL: serverConfig.baseURL)
                    }
                } label: {
                    AppPrimaryActionLabel(
                        title: "Stop",
                        icon: "stop.fill"
                    )
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .frame(maxWidth: .infinity)
            }
            .frame(maxWidth: .infinity)
        }
        .padding(.top, 10)
        .padding(.bottom, 12)
        .frame(width: contentWidth, alignment: .top)
        .background(
            UnevenRoundedRectangle(
                cornerRadii: .init(
                    topLeading: 28,
                    bottomLeading: 0,
                    bottomTrailing: 0,
                    topTrailing: 28
                ),
                style: .continuous
            )
            .fill(Color(uiColor: .systemBackground))
            .shadow(color: Color.black.opacity(0.08), radius: 20, x: 0, y: -8)
        )
        .frame(maxWidth: .infinity, alignment: .center)
    }

    private var sessionOptionsSheet: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: AppSpacing.cardSpacing) {
                    AppCard {
                        AppSectionHeader(
                            title: "Session Options",
                            subtitle: "Profile, backend, and output controls."
                        )

                        VStack(alignment: .leading, spacing: 12) {
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
                .padding(.horizontal, AppSpacing.screenPadding)
                .padding(.top, 16)
                .padding(.bottom, 32)
            }
            .navigationTitle("Session")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        showsSessionOptions = false
                    }
                }
            }
        }
        .presentationDetents([.medium, .large])
    }

    private var diagnosticsSheet: some View {
        NavigationStack {
            ConsoleLogView(logs: viewModel.logs, title: "Stream Logs")
                .padding(16)
                .navigationTitle("Diagnostics")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("Done") {
                            showsDiagnostics = false
                        }
                    }
                }
        }
        .presentationDetents([.medium, .large])
    }

    private var currentBanner: (text: String, tone: AppBannerTone)? {
        if let error = viewModel.errorMessage ?? viewModel.player.errorMessage {
            return (error, .error)
        }
        if let speechError = speech.errorMessage {
            return (speechError, .error)
        }
        return nil
    }

    private func startSpeechIfNeeded() async {
        guard !speech.isRecording, !speech.isStarting else { return }
        if !speech.isAuthorized {
            _ = await speech.requestPermissions()
        }
        await speech.startTranscribing()
    }

    private func finishSpeechAndSend() {
        dismissKeyboard()
        let result: String?
        if speech.isRecording {
            result = speech.stopTranscribing(commitResult: true)
        } else {
            let transcript = speech.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
            result = transcript.isEmpty ? nil : transcript
        }

        if let result {
            inputText = result
            viewModel.endpoint = .chat
            viewModel.startStreaming(baseURL: serverConfig.baseURL, text: result)
        }
    }

    private func dismissKeyboard() {
        inputFocused = false
#if canImport(UIKit)
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
#endif
    }

    private var holdToTalkControl: some View {
        HoldToTalkControl(
            title: speech.isRecording ? "Listening... release to send" : "Hold to Talk",
            icon: speech.isRecording ? "waveform.circle.fill" : "mic.fill",
            isActive: speech.isRecording
        )
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in
                    guard !speechPressIsActive else { return }
                    speechPressIsActive = true
                    dismissKeyboard()
                    Task { await startSpeechIfNeeded() }
                }
                .onEnded { _ in
                    speechPressIsActive = false
                    finishSpeechAndSend()
                }
        )
    }
}

#Preview {
    let session = AppSessionViewModel()
    NavigationStack {
        AvatarChatView()
            .environmentObject(ServerConfig.shared)
            .environmentObject(session)
            .environmentObject(session.streamSession)
    }
}

private struct HoldToTalkControl: View {
    let title: String
    let icon: String
    let isActive: Bool

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
            Text(title)
                .fontWeight(.semibold)
                .lineLimit(1)
                .minimumScaleFactor(0.82)
        }
        .frame(maxWidth: .infinity)
        .frame(height: 52)
        .foregroundStyle(.white)
        .background(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(isActive ? Color.red : Color.blue)
        )
        .contentShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
    }
}
