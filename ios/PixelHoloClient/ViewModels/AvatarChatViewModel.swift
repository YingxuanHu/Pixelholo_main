import Combine
import Foundation

@MainActor
final class AvatarChatViewModel: ObservableObject {
    private static let maxLogLines = 120
    private static let warmupCacheMaxAgeSeconds: TimeInterval = 120
    private static let warmupTimeoutSeconds: UInt64 = 180
    private static let voiceControlsDebounceNanoseconds: UInt64 = 300_000_000

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
    @Published private(set) var voiceControlsStatus: VoiceControlsLoadState = .idle
    @Published private(set) var voiceControlDefaults: VoiceControlValues?
    @Published var voiceControlValues: VoiceControlValues?
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
    private var preparedWarmupAt: Date?
    private var voiceControlBackendDefaults: VoiceControlBackendDefaults?
    private var voiceControlsTask: Task<Void, Never>?
    private var voiceControlsTaskID: UUID?
    private var voiceControlsLoadedKey: String?

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
        clearVoiceControls()

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
        let backend = effectiveWarmupBackend
        let includeLLM = shouldWarmupLLM

        let key = warmupKey(
            baseURL: baseURL,
            profile: cleanedProfile,
            profileType: profileType,
            backend: backend,
            includeLLM: includeLLM
        )
        if hasFreshPreparedWarmup(for: key) {
            Task { [weak self] in
                guard let self else { return }
                if !(await self.hasValidPreparedWarmup(for: key, baseURL: baseURL)) {
                    self.startWarmupTask(
                        baseURL: baseURL,
                        profile: cleanedProfile,
                        profileType: profileType,
                        backend: backend,
                        includeLLM: includeLLM,
                        clearErrors: false
                    )
                }
            }
            return
        }
        if activeWarmupKey == key, warmupTask != nil {
            return
        }

        startWarmupTask(
            baseURL: baseURL,
            profile: cleanedProfile,
            profileType: profileType,
            backend: backend,
            includeLLM: includeLLM,
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
        let streamEndpoint = endpoint
        let warmupBackend = effectiveWarmupBackend
        let includeLLM = streamEndpoint == .chat

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
        if voiceControlsStatus == .ready, let voiceControlValues {
            request.pitchShift = voiceControlValues.pitch
            request.paceScale = resolvePaceScale(voiceControlValues.pace)
            let toneOverrides = resolveToneOverrides(voiceControlValues.tone)
            request.f0Scale = toneOverrides.f0Scale
            request.embeddingScale = toneOverrides.embeddingScale
            request.volumeGain = resolveVolumeGain(voiceControlValues.volume)
        }
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
                backend: warmupBackend,
                includeLLM: includeLLM
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
                    endpoint: streamEndpoint,
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

    func scheduleVoiceControlsRefresh(baseURL: URL?) {
        voiceControlsTask?.cancel()
        voiceControlsTask = Task { [weak self] in
            do {
                try await Task.sleep(nanoseconds: Self.voiceControlsDebounceNanoseconds)
            } catch {
                return
            }
            self?.voiceControlsTask = nil
            await self?.refreshVoiceControlsNow(baseURL: baseURL)
        }
    }

    func refreshVoiceControlsNow(baseURL: URL?) async {
        let cleanedProfile = profileName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let baseURL else {
            clearVoiceControls()
            return
        }
        guard !cleanedProfile.isEmpty else {
            clearVoiceControls()
            return
        }

        let key = voiceControlsKey(baseURL: baseURL, profile: cleanedProfile, profileType: profileType)
        if voiceControlsLoadedKey == key, voiceControlsStatus == .ready {
            return
        }

        voiceControlsTask?.cancel()
        let taskID = UUID()
        voiceControlsTaskID = taskID
        voiceControlsLoadedKey = nil
        voiceControlsStatus = .loading
        voiceControlDefaults = nil
        voiceControlValues = nil
        voiceControlBackendDefaults = nil

        do {
            let response = try await apiClient.fetchVoiceControls(
                baseURL: baseURL,
                profile: cleanedProfile,
                profileType: profileType
            )
            guard voiceControlsTaskID == taskID else { return }
            let backendDefaults = VoiceControlBackendDefaults(
                pitchShift: response.controls.pitchShift ?? 0,
                f0Scale: response.controls.f0Scale ?? 1,
                embeddingScale: response.controls.embeddingScale ?? 1.2,
                paceScale: response.controls.paceScale ?? 1,
                volumeGain: response.controls.volumeGain ?? 1
            )
            let defaults = normalizeVoiceControls(
                VoiceControlValues(
                    pitch: backendDefaults.pitchShift,
                    pace: 0,
                    tone: 0,
                    volume: 0
                )
            )
            voiceControlBackendDefaults = backendDefaults
            voiceControlDefaults = defaults
            voiceControlValues = defaults
            voiceControlsStatus = .ready
            voiceControlsLoadedKey = key
        } catch is CancellationError {
            return
        } catch let error as APIError {
            guard voiceControlsTaskID == taskID else { return }
            switch error {
            case let .server(statusCode, _message) where statusCode == 404:
                voiceControlsStatus = .unavailable(message: "Voice controls appear once the selected profile has a trained model.")
            default:
                voiceControlsStatus = .error(message: error.localizedDescription)
            }
        } catch {
            guard voiceControlsTaskID == taskID else { return }
            voiceControlsStatus = .error(message: error.localizedDescription)
        }
    }

    func updateVoiceControls(_ update: (inout VoiceControlValues) -> Void) {
        guard var values = voiceControlValues ?? voiceControlDefaults else { return }
        update(&values)
        voiceControlValues = normalizeVoiceControls(values)
    }

    func resetVoiceControls() {
        guard let voiceControlDefaults else { return }
        voiceControlValues = voiceControlDefaults
    }

    private var effectiveWarmupBackend: LipsyncBackend? {
        profileType == .avatar ? lipsyncBackend : nil
    }

    private var shouldWarmupLLM: Bool {
        endpoint == .chat
    }

    private func resolvedWarmupAvatarFPS(for backend: LipsyncBackend?) -> Double {
        backend == .wav2lip ? MobileAvatarStreamProfile.wav2lipFPS : MobileAvatarStreamProfile.museTalkFPS
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
        backend: LipsyncBackend?,
        includeLLM: Bool
    ) -> String {
        let backendValue = backend?.rawValue ?? "-"
        let llmValue = includeLLM ? "llm" : "no-llm"
        return "\(baseURL.absoluteString)|\(profileType.rawValue):\(profile):\(backendValue):\(llmValue)"
    }

    private func voiceControlsKey(baseURL: URL, profile: String, profileType: ProfileType) -> String {
        "\(baseURL.absoluteString)|\(profileType.rawValue):\(profile)"
    }

    private func clearPreparedWarmup() {
        preparedWarmupKey = nil
        preparedRuntimeInstanceID = nil
        preparedWarmupAt = nil
    }

    private func clearVoiceControls() {
        voiceControlsTask?.cancel()
        voiceControlsTask = nil
        voiceControlsTaskID = nil
        voiceControlsLoadedKey = nil
        voiceControlBackendDefaults = nil
        voiceControlsStatus = .idle
        voiceControlDefaults = nil
        voiceControlValues = nil
    }

    private func hasFreshPreparedWarmup(for key: String) -> Bool {
        guard preparedWarmupKey == key, let preparedWarmupAt else { return false }
        if Date().timeIntervalSince(preparedWarmupAt) > Self.warmupCacheMaxAgeSeconds {
            clearPreparedWarmup()
            return false
        }
        return true
    }

    private func hasValidPreparedWarmup(for key: String, baseURL: URL) async -> Bool {
        guard hasFreshPreparedWarmup(for: key) else { return false }
        guard let preparedRuntimeInstanceID else {
            clearPreparedWarmup()
            return false
        }
        do {
            let status = try await apiClient.lipsyncBackendStatus(baseURL: baseURL)
            if status.runtimeInstanceID == preparedRuntimeInstanceID {
                return true
            }
        } catch {
            appendLog("warmup status check failed: \(error.localizedDescription)", isError: true)
        }
        clearPreparedWarmup()
        return false
    }

    private func isWarmupComplete(_ response: WarmupResponse, profileType: ProfileType) -> Bool {
        let ttsReady = response.ttsReady ?? (response.ttsHotBefore == true || response.ttsWarmed == true)
        let lipsyncReady = response.lipsyncReady ?? (
            profileType != .avatar || response.lipsyncHotBefore == true || response.lipsyncWarmed == true
        )
        let llmReady = response.llmReady ?? (response.llmHotBefore == true || response.llmWarmed == true)
        return ttsReady && lipsyncReady && (response.llmReady == nil || llmReady)
    }

    private func clamp(_ value: Double, min lowerBound: Double, max upperBound: Double) -> Double {
        min(upperBound, max(lowerBound, value))
    }

    private func normalizeVoiceControls(_ values: VoiceControlValues) -> VoiceControlValues {
        VoiceControlValues(
            pitch: Double(round(clamp(values.pitch, min: -4, max: 4) * 10) / 10),
            pace: Int(clamp(Double(values.pace), min: -100, max: 100).rounded()),
            tone: Int(clamp(Double(values.tone), min: -100, max: 100).rounded()),
            volume: Int(clamp(Double(values.volume), min: -100, max: 100).rounded())
        )
    }

    private func resolvePaceScale(_ pace: Int) -> Double {
        let base = voiceControlBackendDefaults?.paceScale ?? 1
        return Double(round(clamp(base + (Double(pace) / 100) * 0.15, min: 0.85, max: 1.15) * 100) / 100)
    }

    private func resolveToneOverrides(_ tone: Int) -> (f0Scale: Double, embeddingScale: Double) {
        let baseF0 = voiceControlBackendDefaults?.f0Scale ?? 1
        let baseEmbedding = voiceControlBackendDefaults?.embeddingScale ?? 1.2
        let normalized = Double(tone) / 100
        let f0Scale = Double(round(clamp(baseF0 + normalized * 0.18, min: 0.75, max: 1.35) * 100) / 100)
        let embeddingScale = Double(round(clamp(baseEmbedding + normalized * 0.55, min: 0.8, max: 2.2) * 100) / 100)
        return (f0Scale, embeddingScale)
    }

    private func resolveVolumeGain(_ volume: Int) -> Double {
        let base = voiceControlBackendDefaults?.volumeGain ?? 1
        return Double(round(clamp(base + (Double(volume) / 100) * 0.45, min: 0.6, max: 1.45) * 100) / 100)
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
        backend: LipsyncBackend?,
        includeLLM: Bool
    ) async {
        let key = warmupKey(
            baseURL: baseURL,
            profile: profile,
            profileType: profileType,
            backend: backend,
            includeLLM: includeLLM
        )
        if await hasValidPreparedWarmup(for: key, baseURL: baseURL) {
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
            includeLLM: includeLLM,
            clearErrors: true
        )
        await warmupTask?.value
    }

    private func startWarmupTask(
        baseURL: URL,
        profile: String,
        profileType: ProfileType,
        backend: LipsyncBackend?,
        includeLLM: Bool,
        clearErrors: Bool
    ) {
        let key = warmupKey(
            baseURL: baseURL,
            profile: profile,
            profileType: profileType,
            backend: backend,
            includeLLM: includeLLM
        )
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
            force: false,
            includeLLM: includeLLM,
            mobileProfile: true,
            avatarFPS: profileType == .avatar ? resolvedWarmupAvatarFPS(for: backend) : nil,
            musetalkInferFPS: backend == .musetalk ? MobileAvatarStreamProfile.museTalkInferFPS : nil,
            musetalkStreamWindowSec: backend == .musetalk ? MobileAvatarStreamProfile.museTalkWindowSec : nil,
            musetalkLookaheadSec: backend == .musetalk ? MobileAvatarStreamProfile.museTalkLookaheadSec : nil
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
                    if self.isWarmupComplete(response, profileType: profileType) {
                        self.preparedWarmupKey = key
                        self.preparedRuntimeInstanceID = response.runtimeInstanceID
                        self.preparedWarmupAt = Date()
                        self.appendLog("warmup ready \(profile)")
                    } else {
                        self.clearPreparedWarmup()
                        self.appendLog("warmup incomplete \(profile)", isError: true)
                    }
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
