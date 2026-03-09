import SwiftUI

struct AvatarChatView: View {
    @EnvironmentObject private var serverConfig: ServerConfig
    @StateObject private var viewModel = AvatarChatViewModel()
    @StateObject private var speech = SpeechRecognizerManager()
    @State private var inputText = ""

    var body: some View {
        VStack(spacing: 12) {
            Form {
                Section("Session") {
                    TextField("Profile name", text: $viewModel.profileName)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled(true)

                    Picker("Output", selection: $viewModel.profileType) {
                        Text("Voice").tag(ProfileType.voice)
                        Text("Avatar").tag(ProfileType.avatar)
                    }
                    .pickerStyle(.segmented)

                    Picker("Endpoint", selection: $viewModel.endpoint) {
                        Text("Chat").tag(StreamEndpoint.chat)
                        Text("Speak").tag(StreamEndpoint.speak)
                    }
                    .pickerStyle(.segmented)

                    if viewModel.profileType == .avatar {
                        Picker("LipSync Backend", selection: $viewModel.lipsyncBackend) {
                            ForEach(LipsyncBackend.allCases, id: \.self) { backend in
                                Text(backend.rawValue).tag(backend)
                            }
                        }
                    }
                }

                Section("Prompt") {
                    TextEditor(text: $inputText)
                        .frame(minHeight: 96)
                        .textInputAutocapitalization(.sentences)

                    HStack {
                        Button {
                            viewModel.startStreaming(baseURL: serverConfig.baseURL, text: inputText)
                        } label: {
                            Text(viewModel.isStreaming ? "Streaming..." : "Send")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(viewModel.isStreaming)

                        Button {
                            Task {
                                await viewModel.interrupt(baseURL: serverConfig.baseURL)
                            }
                        } label: {
                            Text("Stop / Interrupt")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                    }

                    Button {
                        // gesture-driven button; action intentionally empty
                    } label: {
                        HStack {
                            Image(systemName: speech.isRecording ? "waveform.circle.fill" : "mic.circle")
                            Text(speech.isRecording ? "Listening... release to send" : "Hold to Talk")
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(speech.isRecording ? .red : .blue)
                    .simultaneousGesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { _ in
                                Task { await startSpeechIfNeeded() }
                            }
                            .onEnded { _ in
                                finishSpeechAndSend()
                            }
                    )
                }

                if let error = viewModel.errorMessage ?? viewModel.player.errorMessage {
                    Section("Error") {
                        Text(error).foregroundStyle(.red)
                    }
                }

                if let speechError = speech.errorMessage {
                    Section("Speech") {
                        Text(speechError).foregroundStyle(.red)
                    }
                }
            }

            ZStack {
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(Color.black)
                if let frame = viewModel.player.currentFrame {
                    Image(uiImage: frame)
                        .resizable()
                        .scaledToFill()
                        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                } else {
                    VStack(spacing: 8) {
                        Image(systemName: "person.crop.rectangle")
                            .font(.title2)
                            .foregroundStyle(.white.opacity(0.8))
                        Text(viewModel.isStreaming ? "Waiting for first frame..." : "No frame")
                            .font(.caption)
                            .foregroundStyle(.white.opacity(0.8))
                    }
                }
            }
            .aspectRatio(3.0 / 4.0, contentMode: .fit)
            .padding(.horizontal)

            ConsoleLogView(logs: viewModel.logs, title: "Stream Logs")
                .padding(.horizontal)
        }
        .navigationTitle("Avatar Stream")
        .onDisappear {
            viewModel.stopStreaming()
            _ = speech.stopTranscribing(commitResult: false)
        }
        .task {
            _ = await speech.requestPermissions()
        }
    }

    private func startSpeechIfNeeded() async {
        guard !speech.isRecording else { return }
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
    }
}
