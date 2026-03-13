import Foundation

final class URLSessionLineStreamer: NSObject {
    private struct TaskState {
        var continuation: AsyncThrowingStream<String, Error>.Continuation
        var lineBuffer: Data = Data()
        var statusCode: Int?
        var errorBody: Data = Data()
    }

    private let stateQueue = DispatchQueue(label: "pixelholo.line-streamer.state")
    private lazy var session: URLSession = {
        let config = URLSessionConfiguration.default
        config.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        return URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }()

    private var taskStates: [Int: TaskState] = [:]

    func streamLines(request: URLRequest) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = session.dataTask(with: request)
            let taskID = task.taskIdentifier

            stateQueue.async { [weak self] in
                guard let self else { return }
                self.taskStates[taskID] = TaskState(continuation: continuation)
                task.resume()
            }

            continuation.onTermination = { [weak task, weak self] _ in
                task?.cancel()
                guard let self else { return }
                self.stateQueue.async {
                    self.taskStates.removeValue(forKey: taskID)
                }
            }
        }
    }

    private func processIncomingData(_ data: Data, for taskID: Int) {
        stateQueue.async { [weak self] in
            guard let self else { return }
            guard var state = self.taskStates[taskID] else { return }

            if let code = state.statusCode, !(200...299).contains(code) {
                state.errorBody.append(data)
                self.taskStates[taskID] = state
                return
            }

            state.lineBuffer.append(data)
            let separator = Data([0x0A]) // '\n'

            while let range = state.lineBuffer.range(of: separator) {
                let lineData = state.lineBuffer.subdata(in: 0..<range.lowerBound)
                state.lineBuffer.removeSubrange(0..<range.upperBound)
                let text = String(data: lineData, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if let text, !text.isEmpty {
                    state.continuation.yield(text)
                }
            }

            self.taskStates[taskID] = state
        }
    }

    private func finishTask(_ taskID: Int, error: Error?) {
        stateQueue.async { [weak self] in
            guard let self else { return }
            guard let state = self.taskStates.removeValue(forKey: taskID) else { return }

            if let code = state.statusCode, !(200...299).contains(code) {
                let message = String(data: state.errorBody, encoding: .utf8) ?? "HTTP \(code)"
                state.continuation.finish(throwing: APIError.server(statusCode: code, message: message))
                return
            }

            if !state.lineBuffer.isEmpty {
                let text = String(data: state.lineBuffer, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if let text, !text.isEmpty {
                    state.continuation.yield(text)
                }
            }

            if let error {
                let nsError = error as NSError
                if nsError.domain == NSURLErrorDomain, nsError.code == NSURLErrorCancelled {
                    state.continuation.finish()
                } else {
                    state.continuation.finish(throwing: APIError.transport(error))
                }
            } else {
                state.continuation.finish()
            }
        }
    }
}

extension URLSessionLineStreamer: URLSessionDataDelegate {
    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        stateQueue.async { [weak self] in
            guard let self else { return }
            guard var state = self.taskStates[dataTask.taskIdentifier] else { return }
            if let http = response as? HTTPURLResponse {
                state.statusCode = http.statusCode
            }
            self.taskStates[dataTask.taskIdentifier] = state
            completionHandler(.allow)
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        processIncomingData(data, for: dataTask.taskIdentifier)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        finishTask(task.taskIdentifier, error: error)
    }
}
