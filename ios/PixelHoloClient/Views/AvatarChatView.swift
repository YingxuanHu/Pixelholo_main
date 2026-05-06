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
    @State private var showsVoiceControls = false

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
            await viewModel.refreshVoiceControlsNow(baseURL: serverConfig.baseURL)
        }
        .onChange(of: appSession.selectedProfile?.id) { _, _ in
            viewModel.applySelectedProfile(appSession.selectedProfile)
            viewModel.prepareSelectedProfileWarmup(baseURL: serverConfig.baseURL)
            Task {
                await viewModel.refreshVoiceControlsNow(baseURL: serverConfig.baseURL)
            }
        }
        .onChange(of: viewModel.isStreaming) { _, isStreaming in
            guard !isStreaming else { return }
            viewModel.prepareSelectedProfileWarmup(baseURL: serverConfig.baseURL)
        }
        .onChange(of: viewModel.profileType) { _, _ in
            viewModel.prepareSelectedProfileWarmup(baseURL: serverConfig.baseURL)
            Task {
                await viewModel.refreshVoiceControlsNow(baseURL: serverConfig.baseURL)
            }
        }
        .onChange(of: viewModel.lipsyncBackend) { _, _ in
            viewModel.prepareSelectedProfileWarmup(baseURL: serverConfig.baseURL)
        }
        .onChange(of: viewModel.endpoint) { _, _ in
            viewModel.prepareSelectedProfileWarmup(baseURL: serverConfig.baseURL)
        }
        .onChange(of: serverConfig.baseURL?.absoluteString) { _, _ in
            viewModel.prepareSelectedProfileWarmup(baseURL: serverConfig.baseURL)
            Task {
                await viewModel.refreshVoiceControlsNow(baseURL: serverConfig.baseURL)
            }
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
                                .submitLabel(.done)
                                .onSubmit {
                                    viewModel.prepareSelectedProfileWarmup(baseURL: serverConfig.baseURL)
                                    Task {
                                        await viewModel.refreshVoiceControlsNow(baseURL: serverConfig.baseURL)
                                    }
                                }
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

                    AppCard {
                        DisclosureGroup(isExpanded: $showsVoiceControls) {
                            VStack(alignment: .leading, spacing: 16) {
                                HStack {
                                    Button("Reload Defaults") {
                                        Task {
                                            await viewModel.refreshVoiceControlsNow(baseURL: serverConfig.baseURL)
                                        }
                                    }
                                    .font(.footnote.weight(.semibold))
                                    .buttonStyle(.plain)

                                    Spacer()

                                    Button("Reset") {
                                        viewModel.resetVoiceControls()
                                    }
                                    .font(.footnote.weight(.semibold))
                                    .buttonStyle(.plain)
                                    .disabled(viewModel.voiceControlsStatus != .ready || viewModel.voiceControlDefaults == nil)
                                }

                                voiceControlsContent
                            }
                            .padding(.top, 10)
                        } label: {
                            AppSectionHeader(
                                title: "Voice Controls",
                                subtitle: "Applies to the next response only. Saved profile defaults stay unchanged."
                            )
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

    @ViewBuilder
    private var voiceControlsContent: some View {
        switch viewModel.voiceControlsStatus {
        case .idle:
            Text("Choose or enter a trained profile, then load its defaults here.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        case .loading:
            HStack(spacing: 10) {
                ProgressView()
                Text("Loading voice defaults...")
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(.secondary)
            }
        case let .unavailable(message):
            AppBanner(text: message, tone: .warning)
        case let .error(message):
            AppBanner(text: message, tone: .error)
        case .ready:
            if let values = viewModel.voiceControlValues,
               let defaults = viewModel.voiceControlDefaults {
                VStack(alignment: .leading, spacing: 18) {
                    VoiceControlSliderRow(
                        title: "Pitch",
                        hint: "Make the voice deeper or higher.",
                        valueLabel: describePitch(values.pitch),
                        defaultLabel: describeDefault(describePitch(defaults.pitch)),
                        rangeLabel: "Deeper to Higher"
                    ) {
                        Slider(
                            value: Binding(
                                get: { values.pitch },
                                set: { nextValue in
                                    viewModel.updateVoiceControls { $0.pitch = nextValue }
                                }
                            ),
                            in: -4...4,
                            step: 0.1
                        )
                    }

                    VoiceControlSliderRow(
                        title: "Pace",
                        hint: "Slow the delivery down or speed it up.",
                        valueLabel: describeRelative(values.pace, negative: "slower", positive: "faster"),
                        defaultLabel: describeDefault(describeRelative(defaults.pace, negative: "slower", positive: "faster")),
                        rangeLabel: "Slower to Faster"
                    ) {
                        Slider(
                            value: Binding(
                                get: { Double(values.pace) },
                                set: { nextValue in
                                    viewModel.updateVoiceControls { $0.pace = Int(nextValue.rounded()) }
                                }
                            ),
                            in: -100...100,
                            step: 1
                        )
                    }

                    VoiceControlSliderRow(
                        title: "Tone",
                        hint: "Move the delivery toward calmer or more expressive.",
                        valueLabel: describeRelative(values.tone, negative: "calmer", positive: "more expressive"),
                        defaultLabel: describeDefault(describeRelative(defaults.tone, negative: "calmer", positive: "more expressive")),
                        rangeLabel: "Calmer to More expressive"
                    ) {
                        Slider(
                            value: Binding(
                                get: { Double(values.tone) },
                                set: { nextValue in
                                    viewModel.updateVoiceControls { $0.tone = Int(nextValue.rounded()) }
                                }
                            ),
                            in: -100...100,
                            step: 1
                        )
                    }

                    VoiceControlSliderRow(
                        title: "Volume",
                        hint: "Make the voice softer or stronger.",
                        valueLabel: describeRelative(values.volume, negative: "softer", positive: "stronger"),
                        defaultLabel: describeDefault(describeRelative(defaults.volume, negative: "softer", positive: "stronger")),
                        rangeLabel: "Softer to Stronger"
                    ) {
                        Slider(
                            value: Binding(
                                get: { Double(values.volume) },
                                set: { nextValue in
                                    viewModel.updateVoiceControls { $0.volume = Int(nextValue.rounded()) }
                                }
                            ),
                            in: -100...100,
                            step: 1
                        )
                    }
                }
            }
        }
    }

    private func describePitch(_ value: Double) -> String {
        let amount = abs(value)
        if amount < 0.05 { return "Default" }
        if amount < 1.0 { return value < 0 ? "Slightly deeper" : "Slightly higher" }
        if amount < 2.5 { return value < 0 ? "Deeper" : "Higher" }
        return value < 0 ? "Much deeper" : "Much higher"
    }

    private func describeDefault(_ value: String) -> String {
        value == "Default" ? "Profile default" : "Default \(value.lowercased())"
    }

    private func describeRelative(_ value: Int, negative: String, positive: String) -> String {
        let amount = abs(value)
        if amount < 5 { return "Default" }
        if amount < 35 { return value < 0 ? "Slightly \(negative)" : "Slightly \(positive)" }
        if amount < 70 { return value < 0 ? negative.capitalized : positive.capitalized }
        return value < 0 ? "Much \(negative)" : "Much \(positive)"
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

private struct VoiceControlSliderRow<SliderView: View>: View {
    let title: String
    let hint: String
    let valueLabel: String
    let defaultLabel: String
    let rangeLabel: String
    @ViewBuilder let slider: SliderView

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.subheadline.weight(.semibold))
                    Text(hint)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer(minLength: 8)

                VStack(alignment: .trailing, spacing: 2) {
                    Text(valueLabel)
                        .font(.subheadline.weight(.bold))
                        .foregroundStyle(.tint)
                    Text(defaultLabel)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }

            slider
                .tint(.accentColor)

            Text(rangeLabel)
                .font(.caption2.weight(.medium))
                .foregroundStyle(.secondary)
        }
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
