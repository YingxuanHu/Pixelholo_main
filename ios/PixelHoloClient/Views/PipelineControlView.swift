import SwiftUI

struct PipelineControlView: View {
    @EnvironmentObject private var serverConfig: ServerConfig
    @StateObject private var viewModel = PipelineViewModel()

    @State private var profileName = ""
    @State private var profileType: ProfileType = .avatar

    @State private var batchSize = "2"
    @State private var epochs = "15"
    @State private var maxLen = "400"

    var body: some View {
        VStack(spacing: 12) {
            Form {
                Section("Profile") {
                    TextField("Profile name", text: $profileName)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled(true)
                    Picker("Type", selection: $profileType) {
                        Text("Voice").tag(ProfileType.voice)
                        Text("Avatar").tag(ProfileType.avatar)
                    }
                    .pickerStyle(.segmented)
                }

                Section("Preprocess") {
                    Button {
                        startPreprocess()
                    } label: {
                        Text(viewModel.isPreprocessing ? "Running..." : "Start Preprocess")
                            .frame(maxWidth: .infinity)
                    }
                    .disabled(viewModel.isPreprocessing || profileName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }

                Section("Training") {
                    HStack {
                        Text("Batch")
                        Spacer()
                        TextField("2", text: $batchSize)
                            .multilineTextAlignment(.trailing)
                            .keyboardType(.numberPad)
                            .frame(width: 72)
                    }
                    HStack {
                        Text("Epochs")
                        Spacer()
                        TextField("15", text: $epochs)
                            .multilineTextAlignment(.trailing)
                            .keyboardType(.numberPad)
                            .frame(width: 72)
                    }
                    HStack {
                        Text("Max Len")
                        Spacer()
                        TextField("400", text: $maxLen)
                            .multilineTextAlignment(.trailing)
                            .keyboardType(.numberPad)
                            .frame(width: 72)
                    }
                    Button {
                        startTraining()
                    } label: {
                        Text(viewModel.isTraining ? "Running..." : "Start Training")
                            .frame(maxWidth: .infinity)
                    }
                    .disabled(viewModel.isTraining || profileName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }

                if let error = viewModel.errorMessage {
                    Section("Error") {
                        Text(error).foregroundStyle(.red)
                    }
                }
            }

            ConsoleLogView(logs: viewModel.logs, title: "Backend Logs")
                .padding(.horizontal)

            Button("Stop Job") {
                viewModel.cancelCurrentJob()
            }
            .buttonStyle(.bordered)
            .padding(.bottom, 8)
        }
        .navigationTitle("Pipeline")
    }

    private func startPreprocess() {
        let req = PreprocessRequest(
            profile: profileName.trimmingCharacters(in: .whitespacesAndNewlines),
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
            profile: profileName.trimmingCharacters(in: .whitespacesAndNewlines),
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
}

#Preview {
    NavigationStack {
        PipelineControlView()
            .environmentObject(ServerConfig.shared)
    }
}

