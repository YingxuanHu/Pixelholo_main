import Combine
import Foundation

@MainActor
final class AvatarChatViewModel: ObservableObject {
    private static let maxLogLines = 120

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

    let player: AvatarPlayer

    private let streamingClient: StreamingClient
    private let apiClient: APIClient
    private var streamTask: Task<Void, Never>?
    private var warmupTask: Task<Void, Never>?
    private var activeWarmupKey: String?
    private var warmedProfileKeys: Set<String> = []

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
        profileName = profile.name
        profileType = profile.profileType
    }

    func prepareSelectedProfileWarmup(baseURL: URL?) {
        guard !isStreaming else { return }
        guard let baseURL else { return }
        let cleanedProfile = profileName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanedProfile.isEmpty else { return }

        let key = warmupKey(profile: cleanedProfile, profileType: profileType, backend: effectiveWarmupBackend)
        if warmedProfileKeys.contains(key) || activeWarmupKey == key {
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

        streamTask = Task {
            await ensureWarmupReady(
                baseURL: baseURL,
                profile: cleanedProfile,
                profileType: profileType,
                backend: effectiveWarmupBackend
            )

            if Task.isCancelled {
                await MainActor.run {
                    self.isStreaming = false
                }
                return
            }

            await MainActor.run {
                self.isWarmingUp = false
                self.isStreaming = true
            }

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

            isStreaming = false
            streamTask = nil
        }
    }

    func stopStreaming(clearLogs: Bool = false, cancelWarmup: Bool = true) {
        streamTask?.cancel()
        streamTask = nil
        if cancelWarmup {
            warmupTask?.cancel()
            warmupTask = nil
            activeWarmupKey = nil
        }
        isStreaming = false
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

    private func warmupKey(profile: String, profileType: ProfileType, backend: LipsyncBackend?) -> String {
        let backendValue = backend?.rawValue ?? "-"
        return "\(profileType.rawValue):\(profile):\(backendValue)"
    }

    private func ensureWarmupReady(
        baseURL: URL,
        profile: String,
        profileType: ProfileType,
        backend: LipsyncBackend?
    ) async {
        let key = warmupKey(profile: profile, profileType: profileType, backend: backend)
        if warmedProfileKeys.contains(key) {
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
        let key = warmupKey(profile: profile, profileType: profileType, backend: backend)
        warmupTask?.cancel()
        activeWarmupKey = key
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
                let response = try await self.apiClient.warmup(baseURL: baseURL, request: request)
                if Task.isCancelled { return }
                await MainActor.run {
                    self.warmedProfileKeys.insert(key)
                    if let backendRaw = response.lipsyncBackend,
                       let resolvedBackend = LipsyncBackend(rawValue: backendRaw) {
                        self.lipsyncBackend = resolvedBackend
                    }
                    self.appendLog("warmup ready \(profile)")
                }
            } catch is CancellationError {
                return
            } catch {
                if Task.isCancelled { return }
                await MainActor.run {
                    if clearErrors {
                        self.errorMessage = "Warmup failed: \(error.localizedDescription)"
                    }
                    self.appendLog("warmup error: \(error.localizedDescription)", isError: clearErrors)
                }
            }

            await MainActor.run {
                if self.activeWarmupKey == key {
                    self.activeWarmupKey = nil
                    if !self.isStreaming {
                        self.isWarmingUp = false
                    }
                }
                self.warmupTask = nil
            }
        }
    }
}
