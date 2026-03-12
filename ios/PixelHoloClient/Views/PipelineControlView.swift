import SwiftUI

struct PipelineControlView: View {
    private let contentInset: CGFloat = 14

    @EnvironmentObject private var appSession: AppSessionViewModel
    @EnvironmentObject private var serverConfig: ServerConfig
    @StateObject private var viewModel = PipelineViewModel()
    @StateObject private var profilesViewModel = ProfileListViewModel()

    @State private var profileName = ""
    @State private var profileType: ProfileType = .avatar

    @State private var batchSize = "2"
    @State private var epochs = "15"
    @State private var maxLen = "400"
    @State private var showsSetupSheet = false
    @State private var showsLogsSheet = false

    var body: some View {
        AppScreen {
            ScrollView(.vertical, showsIndicators: false) {
                VStack(spacing: 16) {
                    header

                    if let error = viewModel.errorMessage {
                        AppBanner(text: error, tone: .error)
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
        .sheet(isPresented: $showsLogsSheet) {
            logsSheet
        }
        .task(id: "\(serverConfig.baseURL?.absoluteString ?? "nil")|\(profileType.rawValue)") {
            await profilesViewModel.loadProfiles(baseURL: serverConfig.baseURL, reason: .automatic)
            applySelectedProfile()
        }
        .onChange(of: appSession.selectedProfile?.id) { _, _ in
            applySelectedProfile()
        }
        .onChange(of: appSession.stagedProfileName) { _, _ in
            applySelectedProfile()
        }
        .onChange(of: appSession.stagedProfileType?.rawValue) { _, _ in
            applySelectedProfile()
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
                title: "Train",
                subtitle: "Preprocess media and fine-tune the selected backend profile."
            )

            Spacer(minLength: 12)

            HStack(spacing: 10) {
                AppIconCircleButton(icon: "slider.horizontal.3") {
                    showsSetupSheet = true
                }
                AppIconCircleButton(icon: "arrow.clockwise") {
                    Task {
                        await profilesViewModel.loadProfiles(baseURL: serverConfig.baseURL, reason: .manual)
                    }
                }
                AppIconCircleButton(icon: "text.alignleft") {
                    showsLogsSheet = true
                }
            }
        }
        .padding(.horizontal, contentInset)
    }

    private var workspaceSurface: some View {
        AppPrimarySurface {
            targetSummarySection

            AppSectionDivider()

            actionSection

            if !viewModel.logs.isEmpty {
                AppSectionDivider()
                logPreviewSection
            }
        }
        .padding(.horizontal, contentInset)
    }

    private var targetSummarySection: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(trimmedProfileName.isEmpty ? "No target selected" : trimmedProfileName)
                .font(.system(size: 28, weight: .bold, design: .rounded))
                .foregroundStyle(.primary)

            Picker("Profile Type", selection: $profileType) {
                Text("Voice").tag(ProfileType.voice)
                Text("Avatar").tag(ProfileType.avatar)
            }
            .pickerStyle(.segmented)

            TextField("Type a profile name or tap one below", text: $profileName)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled(true)
                .appInputField()

            if !existingProfilesForCurrentType.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Existing profiles")
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
            } else if serverConfig.baseURL != nil {
                AppBanner(
                    text: profilesViewModel.isLoading ? "Loading profiles..." : "No \(profileType.rawValue) profiles found yet. Create or upload one first.",
                    tone: .neutral
                )
            }

            HStack(spacing: 10) {
                AppMetricPill(
                    title: "Type",
                    value: profileType == .avatar ? "Avatar" : "Voice",
                    tint: profileType == .avatar ? .indigo : .blue
                )
                AppMetricPill(
                    title: "Stage",
                    value: currentStageLabel,
                    tint: viewModel.isTraining || viewModel.isPreprocessing ? .green : .gray
                )
                AppMetricPill(
                    title: "Server",
                    value: serverConfig.baseURL == nil ? "Missing" : "Ready",
                    tint: serverConfig.baseURL == nil ? .orange : .green
                )
            }

            Text("Select the profile you want to preprocess and train. Parameters stay in the setup sheet, but the target stays visible here.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    private var actionSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Run Pipeline")
                .font(.title3.weight(.semibold))

            HStack(spacing: 10) {
                AppMetricPill(title: "Batch", value: batchSize, tint: .blue)
                AppMetricPill(title: "Epochs", value: epochs, tint: .indigo)
                AppMetricPill(title: "Max Len", value: maxLen, tint: .gray)
            }

            if profileType == .avatar {
                Text("Avatar preprocess uses the default bake settings already tuned for the current backend pipeline.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
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
    }

    private var logPreviewSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Recent Backend Activity")
                    .font(.title3.weight(.semibold))
                Spacer()
                Button("Open Logs") {
                    showsLogsSheet = true
                }
                .font(.footnote.weight(.semibold))
            }

            VStack(alignment: .leading, spacing: 8) {
                ForEach(Array(viewModel.logs.suffix(6).enumerated()), id: \.offset) { _, line in
                    Text(line.text)
                        .font(.caption.monospaced())
                        .foregroundStyle(line.isError ? .red : .secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
    }

    private var setupSheet: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: AppSpacing.cardSpacing) {
                    AppCard {
                        AppSectionHeader(
                            title: "Training Parameters",
                            subtitle: "Batch size, epochs, and max length are still configurable here."
                        )

                        VStack(spacing: 12) {
                            trainingNumberField(title: "Batch Size", text: $batchSize)
                            trainingNumberField(title: "Epochs", text: $epochs)
                            trainingNumberField(title: "Max Length", text: $maxLen)
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

    private var logsSheet: some View {
        NavigationStack {
            ConsoleLogView(logs: viewModel.logs, title: "Backend Logs")
                .padding(16)
                .navigationTitle("Logs")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("Done") {
                            showsLogsSheet = false
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

    private var currentStageLabel: String {
        if viewModel.isPreprocessing {
            return "Preprocess"
        }
        if viewModel.isTraining {
            return "Training"
        }
        return "Idle"
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

    private func syncStagedProfile() {
        guard !trimmedProfileName.isEmpty else { return }
        appSession.stageProfileWorkflow(name: trimmedProfileName, type: profileType)
    }

    @ViewBuilder
    private func profileChip(_ profile: ProfileInfo) -> some View {
        let isSelected = trimmedProfileName.caseInsensitiveCompare(profile.name) == .orderedSame
        Button {
            profileName = profile.name
            profileType = profile.profileType
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
        PipelineControlView()
            .environmentObject(ServerConfig.shared)
            .environmentObject(AppSessionViewModel())
    }
}
