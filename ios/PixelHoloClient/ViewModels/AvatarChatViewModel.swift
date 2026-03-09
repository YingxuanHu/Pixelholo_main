import Foundation

@MainActor
final class AvatarChatViewModel: ObservableObject {
    @Published var profileName: String = ""
    @Published var profileType: ProfileType = .avatar
    @Published var endpoint: StreamEndpoint = .chat
    @Published var lipsyncBackend: LipsyncBackend = .musetalk
    @Published var isStreaming = false
    @Published var logs: [ConsoleLogLine] = []
    @Published var errorMessage: String?

    let player: AvatarPlayer

    private let streamingClient: StreamingClient
    private let apiClient: APIClient
    private var streamTask: Task<Void, Never>?

    init(
        player: AvatarPlayer? = nil,
        streamingClient: StreamingClient = StreamingClient(),
        apiClient: APIClient = APIClient()
    ) {
        self.player = player ?? AvatarPlayer()
        self.streamingClient = streamingClient
        self.apiClient = apiClient
    }

    func startStreaming(baseURL: URL?, text: String) {
        guard let baseURL else {
            errorMessage = APIError.invalidBaseURL.localizedDescription
            return
        }
        let cleanedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let cleanedProfile = profileName.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !cleanedText.isEmpty else {
            errorMessage = "Text cannot be empty."
            return
        }
        guard !cleanedProfile.isEmpty else {
            errorMessage = "Profile name is required."
            return
        }

        stopStreaming(clearLogs: false)
        logs.removeAll()
        errorMessage = nil
        player.stop()
        isStreaming = true

        var request = GenerateRequest(
            text: cleanedText,
            speaker: cleanedProfile,
            profileType: profileType,
            avatarProfile: nil,
            lipsyncBackend: nil,
            modelPath: nil,
            refWavPath: nil
        )
        if profileType == .avatar {
            request.avatarProfile = cleanedProfile
            request.lipsyncBackend = lipsyncBackend
        }

        streamTask = Task {
            do {
                let eventStream = try streamingClient.stream(
                    baseURL: baseURL,
                    endpoint: endpoint,
                    request: request
                )

                for try await event in eventStream {
                    switch event {
                    case let .chunk(chunk):
                        player.enqueue(chunk)
                        appendLog("chunk \(chunk.chunkIndex) audio=\(Int(chunk.durationSec * 1000))ms frames=\(chunk.frames.count)")
                    case let .done(inferenceMS):
                        if let inferenceMS {
                            appendLog("done inference_ms=\(Int(inferenceMS))")
                        } else {
                            appendLog("done")
                        }
                    }
                }
            } catch is CancellationError {
                appendLog("stream cancelled")
            } catch {
                errorMessage = error.localizedDescription
                appendLog("stream error: \(error.localizedDescription)", isError: true)
            }

            isStreaming = false
            streamTask = nil
        }
    }

    func stopStreaming(clearLogs: Bool = false) {
        streamTask?.cancel()
        streamTask = nil
        isStreaming = false
        player.stop()
        if clearLogs {
            logs.removeAll()
        }
    }

    func interrupt(baseURL: URL?) async {
        stopStreaming(clearLogs: false)
        guard let baseURL else {
            errorMessage = APIError.invalidBaseURL.localizedDescription
            return
        }

        do {
            let response = try await apiClient.interrupt(baseURL: baseURL)
            appendLog("interrupt: \(response.interruptedStreams) stream(s) stopped")
        } catch {
            errorMessage = error.localizedDescription
            appendLog("interrupt error: \(error.localizedDescription)", isError: true)
        }
    }

    private func appendLog(_ text: String, isError: Bool = false) {
        logs.append(
            ConsoleLogLine(
                timestamp: Date(),
                text: text,
                isError: isError
            )
        )
    }
}
