import SwiftUI
#if canImport(UIKit)
import UIKit
#endif

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
        .onChange(of: viewModel.isStreaming) { _, isStreaming in
            guard !isStreaming else { return }
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
        let stageShape = RoundedRectangle(cornerRadius: previewCornerRadius, style: .continuous)

        return ZStack {
            stageShape
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

            PreviewStageContent(
                player: viewModel.player,
                cornerRadius: previewCornerRadius,
                shouldDisplayFrame: !viewModel.isStreamingDifferentProfile,
                isWarmingCurrentProfile: viewModel.isWarmingCurrentProfile,
                isAwaitingFirstChunkForCurrentProfile: viewModel.isAwaitingFirstChunkForCurrentProfile,
                previewStatusPrimaryColor: previewStatusPrimaryColor,
                previewStatusSecondaryColor: previewStatusSecondaryColor,
                stageEmptyTitle: stageEmptyTitle,
                stageEmptySubtitle: stageEmptySubtitle
            )
        }
        .frame(width: stageWidth, height: stageHeight)
        .frame(maxWidth: .infinity, alignment: .center)
        .clipShape(stageShape)
        .overlay(alignment: .topLeading) {
            if let banner = currentBanner {
                AppBanner(text: banner.text, tone: banner.tone)
                    .padding(.horizontal, 10)
                    .padding(.top, 10)
            }
        }
    }

    private var previewStatusPrimaryColor: Color {
        viewModel.isWarmingCurrentProfile ? .white : .primary
    }

    private var previewStatusSecondaryColor: Color {
        viewModel.isWarmingCurrentProfile ? .white.opacity(0.9) : .secondary
    }

    private var stageEmptyTitle: String {
        if viewModel.isWarmingCurrentProfile {
            let trimmed = viewModel.profileName.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? "Preparing profile" : "Preparing \(trimmed)"
        }
        if viewModel.isAwaitingFirstChunkForCurrentProfile {
            return "Generating"
        }
        if viewModel.isStreamingDifferentProfile {
            return "Another response is playing"
        }
        if viewModel.profileName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "Pick a profile to begin"
        }
        return "Ready to stream"
    }

    private var stageEmptySubtitle: String {
        if viewModel.isWarmingCurrentProfile {
            return "Warming the backend so the first response starts at normal speed."
        }
        if viewModel.isAwaitingFirstChunkForCurrentProfile {
            return "The response is starting. Audio and avatar frames will appear here shortly."
        }
        if viewModel.isStreamingDifferentProfile {
            return "The current audio belongs to a different profile. Stop it or wait for it to finish before switching."
        }
        if viewModel.profileName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "Select a profile from Profiles or enter one in Session Options."
        }
        return "Send a prompt or hold to talk. The avatar preview will appear here."
    }

    private func composerDock(contentWidth: CGFloat) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            FixedTwoLinePromptField(
                text: $inputText,
                placeholder: speech.isRecording
                    ? "Listening..."
                    : "Type something like: Introduce yourself in one sentence."
            )
            .frame(minHeight: 72, alignment: .topLeading)

            holdToTalkControl

            HStack(spacing: 12) {
                Button {
                    dismissKeyboard()
                    viewModel.startStreaming(baseURL: serverConfig.baseURL, text: inputText)
                } label: {
                    AppPrimaryActionLabel(
                        title: viewModel.isWarmingCurrentProfile ? "Preparing..." : (viewModel.isStreaming ? "Streaming..." : "Send"),
                        icon: "paperplane.fill"
                    )
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(viewModel.isStreaming || (viewModel.isWarmingCurrentProfile && inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty))
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

private struct PreviewStageContent: View {
    @ObservedObject var player: AvatarPlayer
    let cornerRadius: CGFloat
    let shouldDisplayFrame: Bool
    let isWarmingCurrentProfile: Bool
    let isAwaitingFirstChunkForCurrentProfile: Bool
    let previewStatusPrimaryColor: Color
    let previewStatusSecondaryColor: Color
    let stageEmptyTitle: String
    let stageEmptySubtitle: String

    var body: some View {
        ZStack {
            if shouldDisplayFrame, player.currentFrame != nil {
                AvatarFrameView(player: player, cornerRadius: cornerRadius)
            }

            if !shouldDisplayFrame || player.currentFrame == nil {
                ZStack(alignment: .center) {
                    if isWarmingCurrentProfile {
                        RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                            .fill(Color.black.opacity(0.18))
                    }

                    VStack(spacing: 12) {
                        if isWarmingCurrentProfile || isAwaitingFirstChunkForCurrentProfile {
                            ProgressView()
                                .progressViewStyle(.circular)
                                .tint(previewStatusPrimaryColor)
                                .scaleEffect(1.15)
                        }

                        Text(stageEmptyTitle)
                            .font(.headline.weight(.semibold))
                            .foregroundStyle(previewStatusPrimaryColor)

                        Text(stageEmptySubtitle)
                            .font(.footnote)
                            .multilineTextAlignment(.center)
                            .foregroundStyle(previewStatusSecondaryColor)
                            .fixedSize(horizontal: false, vertical: true)
                            .padding(.horizontal, 28)
                    }
                    .padding(.horizontal, 24)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
            }
        }
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

private struct FixedTwoLinePromptField: View {
    @Binding var text: String
    let placeholder: String

    var body: some View {
        ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(Color(uiColor: .tertiarySystemBackground))

            AutoScrollingPromptTextView(text: $text)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)

            if text.isEmpty {
                Text(placeholder)
                    .font(.body)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 17)
                    .padding(.vertical, 15)
                    .allowsHitTesting(false)
            }
        }
        .frame(height: 72)
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
    }
}

#if canImport(UIKit)
private struct AutoScrollingPromptTextView: UIViewRepresentable {
    @Binding var text: String

    func makeCoordinator() -> Coordinator {
        Coordinator(text: $text)
    }

    func makeUIView(context: Context) -> UITextView {
        let textView = UITextView()
        textView.delegate = context.coordinator
        textView.backgroundColor = .clear
        textView.font = UIFont.preferredFont(forTextStyle: .body)
        textView.adjustsFontForContentSizeCategory = true
        textView.textColor = UIColor.label
        textView.textContainerInset = .zero
        textView.textContainer.lineFragmentPadding = 0
        textView.textContainer.maximumNumberOfLines = 2
        textView.textContainer.lineBreakMode = .byWordWrapping
        textView.isScrollEnabled = true
        textView.showsVerticalScrollIndicator = false
        textView.showsHorizontalScrollIndicator = false
        textView.keyboardDismissMode = .interactive
        textView.autocapitalizationType = .sentences
        textView.autocorrectionType = .yes
        textView.returnKeyType = .default
        textView.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        textView.text = text
        return textView
    }

    func updateUIView(_ textView: UITextView, context: Context) {
        if textView.text != text {
            context.coordinator.isProgrammaticUpdate = true
            textView.text = text
            context.coordinator.isProgrammaticUpdate = false
        }
        scrollToLatestVisibleText(in: textView)
    }

    private func scrollToLatestVisibleText(in textView: UITextView) {
        guard !textView.text.isEmpty else {
            textView.setContentOffset(.zero, animated: false)
            return
        }
        let lastCharacter = NSRange(location: max(textView.text.utf16.count - 1, 0), length: 1)
        textView.layoutIfNeeded()
        textView.scrollRangeToVisible(lastCharacter)
    }

    final class Coordinator: NSObject, UITextViewDelegate {
        @Binding var text: String
        var isProgrammaticUpdate = false

        init(text: Binding<String>) {
            _text = text
        }

        func textViewDidChange(_ textView: UITextView) {
            guard !isProgrammaticUpdate else { return }
            text = textView.text
            guard !textView.text.isEmpty else { return }
            let lastCharacter = NSRange(location: max(textView.text.utf16.count - 1, 0), length: 1)
            textView.scrollRangeToVisible(lastCharacter)
        }
    }
}
#endif
