import Combine
import Foundation

@MainActor
final class AvatarChatViewModel: ObservableObject {
    private static let maxLogLines = 120
    private static let warmupTimeoutSeconds: UInt64 = 180

    private enum MobileAvatarStreamProfile {
        static let wav2lipFPS = 20.0
        static let wav2lipMaxFrameEdge = 416
        static let wav2lipMaxChunkChars = 96
        static let wav2lipMaxChunkWords = 22
        static let museTalkFPS = 16.0
        static let museTalkInferFPS = 16.0
        static let museTalkWindowSec = 0.42
        static let museTalkLookaheadSec = 0.05
        static let museTalkJPEGQuality = 60
        static let museTalkFirstChunkChars = 48
        static let museTalkMaxChunkChars = 84
        static let museTalkMaxChunkWords = 18
        static let museTalkMaxFrameEdge = 384
    }

    @Published var profileName: String = ""
    @Published var profileType: ProfileType = .avatar
    @Published var endpoint: StreamEndpoint = .chat
    @Published var lipsyncBackend: LipsyncBackend = .musetalk
    @Published var isStreaming = false
    @Published var isWarmingUp = false
    @Published var logs: [ConsoleLogLine] = []
    @Published var errorMessage: String?
    @Published private(set) var warmingProfileStateKey: String?
    @Published private(set) var activeStreamProfileStateKey: String?
    @Published private(set) var awaitingFirstChunkProfileStateKey: String?

    let player: AvatarPlayer

    private let streamingClient: StreamingClient
    private let apiClient: APIClient
    private var streamTask: Task<Void, Never>?
    private var streamTaskID: UUID?
    private var warmupTask: Task<Void, Never>?
    private var warmupTaskID: UUID?
    private var activeWarmupKey: String?
    private var preparedWarmupKey: String?
    private var preparedRuntimeInstanceID: String?

    private enum WarmupFailure: LocalizedError {
        case timedOut(profile: String)

        var errorDescription: String? {
            switch self {
            case let .timedOut(profile):
                return "Preparing \(profile) timed out. The backend may still be building cache data for this profile."
            }
        }
    }

    init(
        initialProfile: ProfileInfo? = nil,
        player: AvatarPlayer? = nil,
        streamingClient: StreamingClient? = nil,
        apiClient: APIClient? = nil
    ) {
        self.player = player ?? AvatarPlayer()
        self.streamingClient = streamingClient ?? StreamingClient()
        self.apiClient = apiClient ?? APIClient()
        if let initialProfile {
            self.profileName = initialProfile.name
            self.profileType = initialProfile.profileType
        }
    }

    func applySelectedProfile(_ profile: ProfileInfo?) {
        guard let profile else { return }
        let previousStateKey = currentProfileStateKey
        profileName = profile.name
        profileType = profile.profileType
        let nextStateKey = currentProfileStateKey
        guard nextStateKey != previousStateKey else { return }

        errorMessage = nil

        if !isStreaming {
            player.stop()
            activeStreamProfileStateKey = nil
            awaitingFirstChunkProfileStateKey = nil
        }

        if warmingProfileStateKey != nextStateKey {
            warmupTask?.cancel()
            warmupTask = nil
            activeWarmupKey = nil
            warmingProfileStateKey = nil
            isWarmingUp = false
        }
    }

    func prepareSelectedProfileWarmup(baseURL: URL?) {
        guard !isStreaming else { return }
        guard let baseURL else { return }
        let cleanedProfile = profileName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanedProfile.isEmpty else { return }

        let key = warmupKey(
            baseURL: baseURL,
            profile: cleanedProfile,
            profileType: profileType,
            backend: effectiveWarmupBackend
        )
        if preparedWarmupKey == key {
            return
        }
        if activeWarmupKey == key, warmupTask != nil {
            return
        }

        startWarmupTask(
            baseURL: baseURL,
            profile: cleanedProfile,
            profileType: profileType,
            backend: effectiveWarmupBackend,
            clearErrors: false
        )
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

        stopStreaming(clearLogs: false, cancelWarmup: false)
        logs.removeAll()
        errorMessage = nil
        player.stop()

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
            if lipsyncBackend == .wav2lip {
                request.avatarFPS = MobileAvatarStreamProfile.wav2lipFPS
                request.avatarMaxFrameEdge = MobileAvatarStreamProfile.wav2lipMaxFrameEdge
                request.maxChunkChars = MobileAvatarStreamProfile.wav2lipMaxChunkChars
                request.maxChunkWords = MobileAvatarStreamProfile.wav2lipMaxChunkWords
            } else {
                request.avatarFPS = MobileAvatarStreamProfile.museTalkFPS
                request.avatarMaxFrameEdge = MobileAvatarStreamProfile.museTalkMaxFrameEdge
                request.maxChunkChars = MobileAvatarStreamProfile.museTalkMaxChunkChars
                request.maxChunkWords = MobileAvatarStreamProfile.museTalkMaxChunkWords
                request.musetalkInferFPS = MobileAvatarStreamProfile.museTalkInferFPS
                request.musetalkStreamWindowSec = MobileAvatarStreamProfile.museTalkWindowSec
                request.musetalkLookaheadSec = MobileAvatarStreamProfile.museTalkLookaheadSec
                request.musetalkJPEGQuality = MobileAvatarStreamProfile.museTalkJPEGQuality
                request.musetalkMaxChunkChars = MobileAvatarStreamProfile.museTalkMaxChunkChars
                request.musetalkFirstChunkChars = MobileAvatarStreamProfile.museTalkFirstChunkChars
            }
        }

        let streamStateKey = profileStateKey(
            profile: cleanedProfile,
            profileType: profileType,
            backend: request.lipsyncBackend
        )
        let taskID = UUID()
        streamTaskID = taskID

        streamTask = Task {
            await ensureWarmupReady(
                baseURL: baseURL,
                profile: cleanedProfile,
                profileType: profileType,
                backend: effectiveWarmupBackend
            )

            if Task.isCancelled {
                if self.streamTaskID == taskID {
                    self.isStreaming = false
                    self.streamTask = nil
                }
                return
            }

            if self.streamTaskID == taskID {
                self.isWarmingUp = false
                self.isStreaming = true
                self.activeStreamProfileStateKey = streamStateKey
                self.awaitingFirstChunkProfileStateKey = streamStateKey
            }

            do {
                let eventStream = try streamingClient.stream(
                    baseURL: baseURL,
                    endpoint: endpoint,
                    request: request
                )

                for try await event in eventStream {
                    if self.streamTaskID != taskID {
                        break
                    }
                    switch event {
                    case let .chunk(chunk):
                        if self.awaitingFirstChunkProfileStateKey == streamStateKey {
                            self.awaitingFirstChunkProfileStateKey = nil
                        }
                        player.enqueue(chunk)
                        appendLog("chunk \(chunk.chunkIndex) audio=\(Int(chunk.durationSec * 1000))ms frames=\(chunk.framePayloads.count)")
                    case let .done(inferenceMS):
                        player.finishStream()
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

            if self.streamTaskID == taskID {
                isStreaming = false
                if self.activeStreamProfileStateKey == streamStateKey {
                    self.activeStreamProfileStateKey = nil
                }
                if self.awaitingFirstChunkProfileStateKey == streamStateKey {
                    self.awaitingFirstChunkProfileStateKey = nil
                }
                streamTask = nil
                streamTaskID = nil
            }
        }
    }

    func stopStreaming(clearLogs: Bool = false, cancelWarmup: Bool = true) {
        streamTask?.cancel()
        streamTask = nil
        streamTaskID = nil
        if cancelWarmup {
            warmupTask?.cancel()
            warmupTask = nil
            warmupTaskID = nil
            activeWarmupKey = nil
            warmingProfileStateKey = nil
        }
        isStreaming = false
        activeStreamProfileStateKey = nil
        awaitingFirstChunkProfileStateKey = nil
        if cancelWarmup {
            isWarmingUp = false
        }
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
        if logs.count > Self.maxLogLines {
            logs.removeFirst(logs.count - Self.maxLogLines)
        }
    }

    private var effectiveWarmupBackend: LipsyncBackend? {
        profileType == .avatar ? lipsyncBackend : nil
    }

    private var currentProfileStateKey: String? {
        let cleanedProfile = profileName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanedProfile.isEmpty else { return nil }
        return profileStateKey(
            profile: cleanedProfile,
            profileType: profileType,
            backend: effectiveWarmupBackend
        )
    }

    var isWarmingCurrentProfile: Bool {
        guard let currentProfileStateKey else { return false }
        return warmingProfileStateKey == currentProfileStateKey
    }

    var isAwaitingFirstChunkForCurrentProfile: Bool {
        guard let currentProfileStateKey else { return false }
        return awaitingFirstChunkProfileStateKey == currentProfileStateKey
    }

    var isStreamingDifferentProfile: Bool {
        guard isStreaming else { return false }
        guard let activeStreamProfileStateKey else { return false }
        guard let currentProfileStateKey else { return true }
        return activeStreamProfileStateKey != currentProfileStateKey
    }

    private func profileStateKey(
        profile: String,
        profileType: ProfileType,
        backend: LipsyncBackend?
    ) -> String {
        let backendValue = backend?.rawValue ?? "-"
        return "\(profileType.rawValue):\(profile):\(backendValue)"
    }

    private func warmupKey(
        baseURL: URL,
        profile: String,
        profileType: ProfileType,
        backend: LipsyncBackend?
    ) -> String {
        let backendValue = backend?.rawValue ?? "-"
        return "\(baseURL.absoluteString)|\(profileType.rawValue):\(profile):\(backendValue)"
    }

    private func performWarmup(baseURL: URL, request: WarmupRequest) async throws -> WarmupResponse {
        try await withThrowingTaskGroup(of: WarmupResponse.self) { group in
            group.addTask { [apiClient] in
                try await apiClient.warmup(baseURL: baseURL, request: request)
            }
            group.addTask {
                try await Task.sleep(nanoseconds: Self.warmupTimeoutSeconds * 1_000_000_000)
                throw WarmupFailure.timedOut(profile: request.profile)
            }
            guard let response = try await group.next() else {
                throw WarmupFailure.timedOut(profile: request.profile)
            }
            group.cancelAll()
            return response
        }
    }

    private func ensureWarmupReady(
        baseURL: URL,
        profile: String,
        profileType: ProfileType,
        backend: LipsyncBackend?
    ) async {
        let key = warmupKey(baseURL: baseURL, profile: profile, profileType: profileType, backend: backend)
        if preparedWarmupKey == key {
            return
        }
        if activeWarmupKey == key, let warmupTask {
            await warmupTask.value
            return
        }

        startWarmupTask(
            baseURL: baseURL,
            profile: profile,
            profileType: profileType,
            backend: backend,
            clearErrors: true
        )
        await warmupTask?.value
    }

    private func startWarmupTask(
        baseURL: URL,
        profile: String,
        profileType: ProfileType,
        backend: LipsyncBackend?,
        clearErrors: Bool
    ) {
        let key = warmupKey(baseURL: baseURL, profile: profile, profileType: profileType, backend: backend)
        let stateKey = profileStateKey(profile: profile, profileType: profileType, backend: backend)
        warmupTask?.cancel()
        let taskID = UUID()
        warmupTaskID = taskID
        activeWarmupKey = key
        warmingProfileStateKey = stateKey
        isWarmingUp = true
        if clearErrors {
            errorMessage = nil
        }
        appendLog("warmup preparing \(profile)")

        let request = WarmupRequest(
            profile: profile,
            profileType: profileType,
            lipsyncBackend: backend,
            force: false
        )

        warmupTask = Task { [weak self] in
            guard let self else { return }
            do {
                let response = try await self.performWarmup(baseURL: baseURL, request: request)
                if Task.isCancelled { return }
                if self.warmupTaskID == taskID {
                    if let backendRaw = response.lipsyncBackend,
                       let resolvedBackend = LipsyncBackend(rawValue: backendRaw) {
                        self.lipsyncBackend = resolvedBackend
                    }
                    self.preparedWarmupKey = key
                    self.preparedRuntimeInstanceID = response.runtimeInstanceID
                    self.appendLog("warmup ready \(profile)")
                }
            } catch is CancellationError {
                return
            } catch {
                if Task.isCancelled { return }
                if self.warmupTaskID == taskID {
                    if self.warmingProfileStateKey == stateKey {
                        self.errorMessage = "Warmup failed: \(error.localizedDescription)"
                    }
                    self.appendLog("warmup error: \(error.localizedDescription)", isError: true)
                }
            }

            if self.warmupTaskID == taskID {
                if self.activeWarmupKey == key {
                    self.activeWarmupKey = nil
                    self.warmingProfileStateKey = nil
                    self.isWarmingUp = false
                }
                self.warmupTask = nil
                self.warmupTaskID = nil
            }
        }
    }
}
