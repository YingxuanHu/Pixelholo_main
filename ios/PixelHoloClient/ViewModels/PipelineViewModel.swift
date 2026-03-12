import Combine
import Foundation

@MainActor
final class PipelineViewModel: ObservableObject {
    @Published var logs: [ConsoleLogLine] = []
    @Published var isPreprocessing = false
    @Published var isTraining = false
    @Published var errorMessage: String?

    private let apiClient: APIClient
    private var activeTask: Task<Void, Never>?

    init(apiClient: APIClient? = nil) {
        self.apiClient = apiClient ?? APIClient()
    }

    func startPreprocess(baseURL: URL?, request: PreprocessRequest) {
        guard let baseURL else {
            errorMessage = APIError.invalidBaseURL.localizedDescription
            return
        }
        cancelCurrentJob()
        logs = []
        errorMessage = nil
        isPreprocessing = true
        isTraining = false

        activeTask = Task {
            do {
                let lineStream = try apiClient.startPreprocess(baseURL: baseURL, request: request)
                for try await line in lineStream {
                    appendLog(line)
                }
            } catch {
                appendLog("Preprocess failed: \(error.localizedDescription)", isError: true)
                errorMessage = error.localizedDescription
            }
            isPreprocessing = false
            activeTask = nil
        }
    }

    func startTraining(baseURL: URL?, request: TrainRequest) {
        guard let baseURL else {
            errorMessage = APIError.invalidBaseURL.localizedDescription
            return
        }
        cancelCurrentJob()
        logs = []
        errorMessage = nil
        isTraining = true
        isPreprocessing = false

        activeTask = Task {
            do {
                let lineStream = try apiClient.startTraining(baseURL: baseURL, request: request)
                for try await line in lineStream {
                    appendLog(line)
                }
            } catch {
                appendLog("Training failed: \(error.localizedDescription)", isError: true)
                errorMessage = error.localizedDescription
            }
            isTraining = false
            activeTask = nil
        }
    }

    func cancelCurrentJob() {
        activeTask?.cancel()
        activeTask = nil
        isPreprocessing = false
        isTraining = false
    }

    private func appendLog(_ text: String, isError: Bool? = nil) {
        let inferredError: Bool
        if let isError {
            inferredError = isError
        } else {
            let lowered = text.lowercased()
            inferredError = lowered.contains("error") || lowered.contains("failed") || lowered.contains("traceback")
        }
        logs.append(
            ConsoleLogLine(
                timestamp: Date(),
                text: text,
                isError: inferredError
            )
        )
    }
}
