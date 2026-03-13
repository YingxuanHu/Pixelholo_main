import SwiftUI

struct ProfileListView: View {
    private let contentInset: CGFloat = 14

    @EnvironmentObject private var appSession: AppSessionViewModel
    @EnvironmentObject private var serverConfig: ServerConfig
    @StateObject private var viewModel = ProfileListViewModel()
    @State private var showsServerSheet = false

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
        .sheet(isPresented: $showsServerSheet) {
            serverSettingsSheet
        }
        .task(id: serverConfig.baseURL?.absoluteString) {
            await viewModel.loadProfiles(baseURL: serverConfig.baseURL, reason: .automatic)
        }
        .refreshable {
            await viewModel.loadProfiles(baseURL: serverConfig.baseURL, reason: .manual)
        }
    }

    private var header: some View {
        HStack(alignment: .top, spacing: 14) {
            AppScreenHeader(
                title: "Profiles",
                subtitle: "Browse voice and avatar profiles from the active backend."
            )

            Spacer(minLength: 12)

            HStack(spacing: 10) {
                AppIconCircleButton(icon: "slider.horizontal.3") {
                    showsServerSheet = true
                }
                AppIconCircleButton(icon: viewModel.isLoading ? "hourglass" : "arrow.clockwise") {
                    Task {
                        await viewModel.loadProfiles(baseURL: serverConfig.baseURL, reason: .manual)
                    }
                }
            }
        }
        .padding(.horizontal, contentInset)
    }

    private var workspaceSurface: some View {
        AppPrimarySurface {
            backendStrip

            AppSectionDivider()

            profileGroup(
                title: "Voice Profiles",
                subtitle: "Speech-only models for direct audio output.",
                profiles: viewModel.voiceProfiles,
                emptyMessage: "No voice profiles found."
            )

            AppSectionDivider()

            profileGroup(
                title: "Avatar Profiles",
                subtitle: "Profiles with baked avatar assets for lip-sync playback.",
                profiles: viewModel.avatarProfiles,
                emptyMessage: "No avatar profiles found."
            )
        }
        .padding(.horizontal, contentInset)
    }

    private var backendStrip: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 5) {
                    Text("Backend")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    Text(serverConfig.baseURLString.isEmpty ? "No server configured" : serverConfig.baseURLString)
                        .font(.subheadline.weight(.medium))
                        .lineLimit(2)
                        .textSelection(.enabled)
                }

                Spacer(minLength: 12)

                if viewModel.isLoading {
                    ProgressView()
                }
            }

            HStack(spacing: 10) {
                AppMetricPill(title: "Voice", value: "\(viewModel.voiceProfiles.count)", tint: .blue)
                AppMetricPill(title: "Avatar", value: "\(viewModel.avatarProfiles.count)", tint: .indigo)
                AppMetricPill(title: "Total", value: "\(viewModel.voiceProfiles.count + viewModel.avatarProfiles.count)", tint: .gray)
            }

            Text(serverConfig.usesLoopbackHost
                 ? "127.0.0.1 only works when the backend is on this same Mac. Use the VM or Tailscale address instead."
                 : "Pull to refresh or use the top-right refresh button when profiles change.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private func profileGroup(
        title: String,
        subtitle: String,
        profiles: [ProfileInfo],
        emptyMessage: String
    ) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(alignment: .firstTextBaseline) {
                    Text(title)
                        .font(.title3.weight(.semibold))
                    Text("(\(profiles.count))")
                        .font(.title3.weight(.semibold))
                        .foregroundStyle(.secondary)
                }
                Text(subtitle)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            if profiles.isEmpty {
                Text(emptyMessage)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .padding(.vertical, 12)
            } else {
                VStack(spacing: 0) {
                    ForEach(Array(profiles.enumerated()), id: \.element.id) { index, profile in
                        Button {
                            appSession.openProfileForStreaming(profile)
                        } label: {
                            ProfileRow(profile: profile)
                                .padding(.horizontal, 4)
                        }
                        .buttonStyle(.plain)

                        if index < profiles.count - 1 {
                            Divider()
                                .padding(.leading, 4)
                        }
                    }
                }
            }
        }
    }

    private var serverSettingsSheet: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    AppCard {
                        AppSectionHeader(
                            title: "Backend Connection",
                            subtitle: "Set the reachable PixelHolo HTTP address."
                        )

                        TextField("http://100.120.224.119:8000", text: $serverConfig.baseURLString)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled(true)
                            .keyboardType(.URL)
                            .submitLabel(.go)
                            .onSubmit {
                                Task {
                                    await viewModel.loadProfiles(baseURL: serverConfig.baseURL, reason: .manual)
                                }
                            }
                            .appInputField()

                        Text("Use the base HTTP address only. Example: http://100.120.224.119:8000")
                            .font(.footnote)
                            .foregroundStyle(.secondary)

                        Button {
                            Task {
                                await viewModel.loadProfiles(baseURL: serverConfig.baseURL, reason: .manual)
                            }
                        } label: {
                            AppPrimaryActionLabel(
                                title: viewModel.isLoading ? "Loading Profiles..." : "Load Profiles",
                                icon: "arrow.clockwise"
                            )
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(viewModel.isLoading)
                    }
                }
                .padding(.horizontal, AppSpacing.screenPadding)
                .padding(.top, 16)
                .padding(.bottom, 32)
            }
            .navigationTitle("Server")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        showsServerSheet = false
                    }
                }
            }
        }
        .presentationDetents([.medium, .large])
    }
}

private struct ProfileRow: View {
    let profile: ProfileInfo

    var body: some View {
        HStack(alignment: .top, spacing: 14) {
            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .center, spacing: 8) {
                    Text(profile.name)
                        .font(.headline)
                        .foregroundStyle(.primary)

                    ProfileStateBadge(isTrained: profile.hasProfile)
                }

                HStack(spacing: 8) {
                    ProfileMetricBadge(label: "Clips", value: "\(profile.processedWavs)")
                    ProfileMetricBadge(label: "Audio", value: "\(profile.rawAudioFiles)")
                    ProfileMetricBadge(label: "Video", value: "\(profile.rawFiles)")
                }

                Text(profile.profileType == .avatar ? "Opens avatar streaming." : "Opens voice streaming.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 12)

            Image(systemName: "arrow.up.right.circle.fill")
                .font(.title3)
                .foregroundStyle(.secondary)
                .padding(.top, 2)
        }
        .contentShape(Rectangle())
        .padding(.vertical, 14)
    }
}

private struct ProfileStateBadge: View {
    let isTrained: Bool

    var body: some View {
        Text(isTrained ? "Trained" : "Needs Training")
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(
                Capsule(style: .continuous)
                    .fill((isTrained ? Color.green : Color.orange).opacity(0.14))
            )
            .foregroundStyle(isTrained ? .green : .orange)
    }
}

private struct ProfileMetricBadge: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(value)
                .font(.subheadline.weight(.semibold))
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color(uiColor: .tertiarySystemBackground))
        )
    }
}

#Preview {
    NavigationStack {
        ProfileListView()
            .environmentObject(ServerConfig.shared)
            .environmentObject(AppSessionViewModel())
    }
}
