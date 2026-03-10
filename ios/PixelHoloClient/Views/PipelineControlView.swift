import SwiftUI

struct PipelineControlView: View {
    @EnvironmentObject private var appSession: AppSessionViewModel
    @EnvironmentObject private var serverConfig: ServerConfig
    @StateObject private var viewModel = PipelineViewModel()

    @State private var profileName = ""
    @State private var profileType: ProfileType = .avatar

    @State private var batchSize = "2"
    @State private var epochs = "15"
    @State private var maxLen = "400"

    var body: some View {
        AppScreen {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: AppSpacing.cardSpacing) {
                    AppScreenHeader(
                        title: "Train",
                        subtitle: "Run preprocess and fine-tuning steps against the selected backend profile."
                    )

                    AppCard {
                        AppSectionHeader(
                            title: "Training Target",
                            subtitle: "Pick the profile you want to preprocess or train. Tapping a profile in the Profiles tab will prefill this screen."
                        )

                        TextField("Profile name", text: $profileName)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled(true)
                            .appInputField()

                        VStack(alignment: .leading, spacing: 8) {
                            Text("Profile Type")
                                .font(.subheadline.weight(.medium))
                                .foregroundStyle(.secondary)

                            Picker("Type", selection: $profileType) {
                                Text("Voice").tag(ProfileType.voice)
                                Text("Avatar").tag(ProfileType.avatar)
                            }
                            .pickerStyle(.segmented)
                        }
                    }

                    AppCard {
                        AppSectionHeader(
                            title: "Stage 1: Preprocess",
                            subtitle: "Extract audio, segment clips, generate metadata, and bake avatar assets when needed."
                        )

                        if profileType == .avatar {
                            AppBanner(
                                text: "Avatar preprocess uses the default bake settings tuned for the current backend pipeline.",
                                tone: .neutral
                            )
                        }

                        Button {
                            startPreprocess()
                        } label: {
                            AppPrimaryActionLabel(
                                title: viewModel.isPreprocessing ? "Preprocessing..." : "Start Preprocess",
                                icon: "wand.and.stars"
                            )
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(viewModel.isPreprocessing || trimmedProfileName.isEmpty)
                    }

                    AppCard {
                        AppSectionHeader(
                            title: "Stage 2: Train",
                            subtitle: "Fine-tune the StyleTTS2 profile after preprocess finishes cleanly."
                        )

                        VStack(spacing: 12) {
                            trainingNumberField(title: "Batch Size", text: $batchSize)
                            trainingNumberField(title: "Epochs", text: $epochs)
                            trainingNumberField(title: "Max Length", text: $maxLen)
                        }

                        Button {
                            startTraining()
                        } label: {
                            AppPrimaryActionLabel(
                                title: viewModel.isTraining ? "Training..." : "Start Training",
                                icon: "bolt.fill"
                            )
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(viewModel.isTraining || trimmedProfileName.isEmpty)

                        Button {
                            viewModel.cancelCurrentJob()
                        } label: {
                            AppPrimaryActionLabel(title: "Stop Current Job", icon: "stop.fill")
                        }
                        .buttonStyle(.bordered)
                    }

                    if let error = viewModel.errorMessage {
                        AppBanner(text: error, tone: .error)
                    }

                    AppCard {
                        ConsoleLogView(logs: viewModel.logs, title: "Backend Logs")
                    }
                }
                .padding(.horizontal, AppSpacing.screenPadding)
                .padding(.top, 12)
                .padding(.bottom, AppSpacing.bottomTabClearance)
            }
        }
        .toolbar(.hidden, for: .navigationBar)
        .task {
            applySelectedProfile()
        }
        .onChange(of: appSession.selectedProfile?.id) { _, _ in
            applySelectedProfile()
        }
    }

    private var trimmedProfileName: String {
        profileName.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func trainingNumberField(title: String, text: Binding<String>) -> some View {
        AppKeyValueRow(title) {
            TextField("0", text: text)
                .multilineTextAlignment(.trailing)
                .keyboardType(.numberPad)
                .frame(width: 88)
                .appInputField()
        }
    }

    private func startPreprocess() {
        let req = PreprocessRequest(
            profile: trimmedProfileName,
            filename: nil,
            audioFilename: nil,
            profileType: profileType,
            bakeAvatar: profileType == .avatar,
            avatarFPS: profileType == .avatar ? 25 : nil,
            avatarStartSec: profileType == .avatar ? 5 : nil,
            avatarLoopSec: profileType == .avatar ? 10 : nil,
            avatarLoopFadeSec: nil,
            avatarResizeFactor: profileType == .avatar ? 1 : nil,
            avatarPads: profileType == .avatar ? "0 10 0 0" : nil,
            avatarBatchSize: nil,
            avatarNosmooth: false,
            avatarBlurBackground: profileType == .avatar ? true : nil,
            avatarBlurKernel: profileType == .avatar ? 75 : nil,
            avatarDevice: nil
        )
        viewModel.startPreprocess(baseURL: serverConfig.baseURL, request: req)
    }

    private func startTraining() {
        let req = TrainRequest(
            profile: trimmedProfileName,
            profileType: profileType,
            batchSize: Int(batchSize),
            epochs: Int(epochs),
            maxLen: Int(maxLen),
            autoSelectEpoch: true,
            autoTuneProfile: true,
            autoBuildLexicon: true,
            selectThorough: true,
            earlyStop: true
        )
        viewModel.startTraining(baseURL: serverConfig.baseURL, request: req)
    }

    private func applySelectedProfile() {
        guard let profile = appSession.selectedProfile else { return }
        profileName = profile.name
        profileType = profile.profileType
    }
}

#Preview {
    NavigationStack {
        PipelineControlView()
            .environmentObject(ServerConfig.shared)
            .environmentObject(AppSessionViewModel())
    }
}
