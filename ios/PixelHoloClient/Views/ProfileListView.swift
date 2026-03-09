import SwiftUI

struct ProfileListView: View {
    @EnvironmentObject private var serverConfig: ServerConfig
    @StateObject private var viewModel = ProfileListViewModel()

    var body: some View {
        List {
            Section("Server") {
                VStack(alignment: .leading, spacing: 8) {
                    TextField("http://192.168.1.x:8000", text: $serverConfig.baseURLString)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled(true)
                        .keyboardType(.URL)
                    Button {
                        Task {
                            await viewModel.loadProfiles(baseURL: serverConfig.baseURL)
                        }
                    } label: {
                        Text(viewModel.isLoading ? "Loading..." : "Load Profiles")
                    }
                    .disabled(viewModel.isLoading)
                }
            }

            if let error = viewModel.errorMessage {
                Section("Error") {
                    Text(error)
                        .foregroundStyle(.red)
                }
            }

            Section("Voice Profiles (\(viewModel.voiceProfiles.count))") {
                if viewModel.voiceProfiles.isEmpty {
                    Text("No voice profiles found.")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(viewModel.voiceProfiles) { profile in
                        ProfileRow(profile: profile)
                    }
                }
            }

            Section("Avatar Profiles (\(viewModel.avatarProfiles.count))") {
                if viewModel.avatarProfiles.isEmpty {
                    Text("No avatar profiles found.")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(viewModel.avatarProfiles) { profile in
                        ProfileRow(profile: profile)
                    }
                }
            }
        }
        .navigationTitle("PixelHolo Profiles")
        .toolbar {
            ToolbarItemGroup(placement: .topBarTrailing) {
                NavigationLink("Stream") {
                    AvatarChatView()
                }
                NavigationLink("Pipeline") {
                    PipelineControlView()
                }
                NavigationLink("Create") {
                    CreateProfileView()
                }
            }
        }
        .task {
            await viewModel.loadProfiles(baseURL: serverConfig.baseURL)
        }
        .refreshable {
            await viewModel.loadProfiles(baseURL: serverConfig.baseURL)
        }
    }
}

private struct ProfileRow: View {
    let profile: ProfileInfo

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(profile.name)
                .font(.headline)
            Text("\(profile.processedWavs) clips • \(profile.rawAudioFiles) audio • \(profile.rawFiles) video")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(profile.hasProfile ? "Trained: yes" : "Trained: no")
                .font(.caption2)
                .foregroundStyle(profile.hasProfile ? .green : .orange)
        }
        .padding(.vertical, 2)
    }
}

#Preview {
    NavigationStack {
        ProfileListView()
            .environmentObject(ServerConfig.shared)
    }
}
