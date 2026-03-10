import SwiftUI

struct ProfileListView: View {
    @EnvironmentObject private var appSession: AppSessionViewModel
    @EnvironmentObject private var serverConfig: ServerConfig
    @StateObject private var viewModel = ProfileListViewModel()

    var body: some View {
        AppScreen {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: AppSpacing.cardSpacing) {
                    AppScreenHeader(
                        title: "Profiles",
                        subtitle: "Load voice and avatar profiles from your active PixelHolo backend."
                    )

                    AppCard {
                        AppSectionHeader(
                            title: "Backend Connection",
                            subtitle: "Enter the reachable PixelHolo server address and load profiles."
                        )

                        TextField("http://100.120.224.119:8000", text: $serverConfig.baseURLString)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled(true)
                            .keyboardType(.URL)
                            .submitLabel(.go)
                            .onSubmit {
                                Task {
                                    await viewModel.loadProfiles(baseURL: serverConfig.baseURL)
                                }
                            }
                            .appInputField()

                        Text(serverConfig.usesLoopbackHost
                             ? "127.0.0.1 only works when the backend runs on this same Mac. For your VM or another machine, use its reachable IP or Tailscale address."
                             : "Use the base HTTP address only. Example: http://100.120.224.119:8000")
                            .font(.footnote)
                            .foregroundStyle(.secondary)

                        Button {
                            Task {
                                await viewModel.loadProfiles(baseURL: serverConfig.baseURL)
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

                    if let error = viewModel.errorMessage {
                        AppBanner(text: error, tone: .error)
                    }

                    profileSection(
                        title: "Voice Profiles (\(viewModel.voiceProfiles.count))",
                        subtitle: "Speech-only models for direct audio output.",
                        profiles: viewModel.voiceProfiles,
                        emptyMessage: "No voice profiles found."
                    )

                    profileSection(
                        title: "Avatar Profiles (\(viewModel.avatarProfiles.count))",
                        subtitle: "Profiles with baked avatar assets for lip-sync playback.",
                        profiles: viewModel.avatarProfiles,
                        emptyMessage: "No avatar profiles found."
                    )
                }
                .padding(.horizontal, AppSpacing.screenPadding)
                .padding(.top, 12)
                .padding(.bottom, AppSpacing.bottomTabClearance)
            }
        }
        .toolbar(.hidden, for: .navigationBar)
        .task {
            await viewModel.loadProfiles(baseURL: serverConfig.baseURL)
        }
        .refreshable {
            await viewModel.loadProfiles(baseURL: serverConfig.baseURL)
        }
    }

    @ViewBuilder
    private func profileSection(
        title: String,
        subtitle: String,
        profiles: [ProfileInfo],
        emptyMessage: String
    ) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            AppSectionHeader(title: title, subtitle: subtitle)

            if profiles.isEmpty {
                AppCard {
                    Text(emptyMessage)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            } else {
                AppCard {
                    VStack(spacing: 0) {
                        ForEach(Array(profiles.enumerated()), id: \.element.id) { index, profile in
                            Button {
                                appSession.openProfileForStreaming(profile)
                            } label: {
                                ProfileRow(profile: profile)
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

                Text(profile.profileType == .avatar ? "Opens avatar stream controls." : "Opens voice stream controls.")
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
