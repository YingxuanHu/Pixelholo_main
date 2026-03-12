import Foundation

final class URLSessionBinaryStreamer: NSObject {
    private struct TaskState {
        var continuation: AsyncThrowingStream<Data, Error>.Continuation
        var statusCode: Int?
        var errorBody = Data()
    }

    private let stateQueue = DispatchQueue(label: "pixelholo.binary-streamer.state")
    private lazy var session: URLSession = {
        let config = URLSessionConfiguration.default
        config.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        return URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }()

    private var taskStates: [Int: TaskState] = [:]

    func streamData(request: URLRequest) -> AsyncThrowingStream<Data, Error> {
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

            state.continuation.yield(data)
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

extension URLSessionBinaryStreamer: URLSessionDataDelegate {
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

