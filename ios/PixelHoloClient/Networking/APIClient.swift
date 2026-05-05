import Foundation
import UniformTypeIdentifiers

final class APIClient {
    private let session: URLSession
    private let decoder: JSONDecoder
    private let lineStreamer: URLSessionLineStreamer

    init(
        session: URLSession = .shared,
        lineStreamer: URLSessionLineStreamer = URLSessionLineStreamer()
    ) {
        self.session = session
        self.decoder = JSONDecoder()
        self.lineStreamer = lineStreamer
    }

    func fetchProfiles(
        baseURL: URL,
        profileType: ProfileType? = nil
    ) async throws -> [ProfileInfo] {
        let endpoint = baseURL.appendingPathComponent("profiles")
        var components = URLComponents(url: endpoint, resolvingAgainstBaseURL: false)
        if let profileType {
            components?.queryItems = [URLQueryItem(name: "profile_type", value: profileType.rawValue)]
        }
        guard let finalURL = components?.url else {
            throw APIError.invalidBaseURL
        }

        var request = URLRequest(url: finalURL)
        request.httpMethod = "GET"
        request.timeoutInterval = 20
        request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData

        let decoded: ProfilesResponse = try await decodedResponse(for: request)
        return decoded.profiles
    }

    func uploadVideo(
        baseURL: URL,
        fileURL: URL,
        profile: String,
        profileType: ProfileType
    ) async throws -> UploadResponse {
        try await uploadFile(
            baseURL: baseURL,
            endpoint: "upload",
            fileURL: fileURL,
            profile: profile,
            profileType: profileType
        )
    }

    func uploadAudio(
        baseURL: URL,
        fileURL: URL,
        profile: String,
        profileType: ProfileType
    ) async throws -> UploadResponse {
        try await uploadFile(
            baseURL: baseURL,
            endpoint: "upload_audio",
            fileURL: fileURL,
            profile: profile,
            profileType: profileType
        )
    }

    private func uploadFile(
        baseURL: URL,
        endpoint: String,
        fileURL: URL,
        profile: String,
        profileType: ProfileType
    ) async throws -> UploadResponse {
        let endpointURL = baseURL.appendingPathComponent(endpoint)
        var request = URLRequest(url: endpointURL)
        request.httpMethod = "POST"
        request.timeoutInterval = 120

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        let body = try makeMultipartBody(
            fileURL: fileURL,
            profile: profile,
            profileType: profileType,
            boundary: boundary
        )

        return try await decodedUploadResponse(for: request, body: body)
    }

    private func makeMultipartBody(
        fileURL: URL,
        profile: String,
        profileType: ProfileType,
        boundary: String
    ) throws -> Data {
        var body = Data()
        let lineBreak = "\r\n"

        func appendField(name: String, value: String) {
            body.append("--\(boundary)\(lineBreak)")
            body.append("Content-Disposition: form-data; name=\"\(name)\"\(lineBreak + lineBreak)")
            body.append("\(value)\(lineBreak)")
        }

        appendField(name: "profile", value: profile)
        appendField(name: "profile_type", value: profileType.rawValue)

        let fileData = try Data(contentsOf: fileURL)
        let filename = fileURL.lastPathComponent
        let mimeType = Self.mimeType(for: fileURL.pathExtension)

        body.append("--\(boundary)\(lineBreak)")
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(filename)\"\(lineBreak)")
        body.append("Content-Type: \(mimeType)\(lineBreak + lineBreak)")
        body.append(fileData)
        body.append(lineBreak)
        body.append("--\(boundary)--\(lineBreak)")

        return body
    }

    private static func mimeType(for fileExtension: String) -> String {
        let ext = fileExtension.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !ext.isEmpty else {
            return "application/octet-stream"
        }
        return UTType(filenameExtension: ext)?.preferredMIMEType ?? "application/octet-stream"
    }

    func startPreprocess(
        baseURL: URL,
        request: PreprocessRequest
    ) throws -> AsyncThrowingStream<String, Error> {
        let urlRequest = try makeJSONRequest(
            baseURL: baseURL,
            endpoint: "preprocess",
            timeout: 60 * 60,
            body: request
        )
        return lineStreamer.streamLines(request: urlRequest)
    }

    func startTraining(
        baseURL: URL,
        request: TrainRequest
    ) throws -> AsyncThrowingStream<String, Error> {
        let urlRequest = try makeJSONRequest(
            baseURL: baseURL,
            endpoint: "train",
            timeout: 60 * 60,
            body: request
        )
        return lineStreamer.streamLines(request: urlRequest)
    }

    func interrupt(baseURL: URL) async throws -> InterruptResponse {
        let endpointURL = baseURL.appendingPathComponent("interrupt")
        var request = URLRequest(url: endpointURL)
        request.httpMethod = "POST"
        request.timeoutInterval = 15

        return try await decodedResponse(for: request)
    }

    func warmup(baseURL: URL, request warmupRequest: WarmupRequest) async throws -> WarmupResponse {
        let request = try makeJSONRequest(
            baseURL: baseURL,
            endpoint: "warmup",
            timeout: 190,
            body: warmupRequest
        )
        return try await decodedResponse(for: request)
    }

    func fetchVoiceControls(
        baseURL: URL,
        profile: String,
        profileType: ProfileType
    ) async throws -> ProfileVoiceControlsResponse {
        let endpointURL = baseURL
            .appendingPathComponent("profiles")
            .appendingPathComponent(profile)
            .appendingPathComponent("voice-controls")
        var components = URLComponents(url: endpointURL, resolvingAgainstBaseURL: false)
        components?.queryItems = [URLQueryItem(name: "profile_type", value: profileType.rawValue)]
        guard let finalURL = components?.url else {
            throw APIError.invalidBaseURL
        }

        var request = URLRequest(url: finalURL)
        request.httpMethod = "GET"
        request.timeoutInterval = 20
        request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData

        return try await decodedResponse(for: request)
    }

    func lipsyncBackendStatus(baseURL: URL) async throws -> LipsyncBackendStatusResponse {
        let endpointURL = baseURL.appendingPathComponent("lipsync_backend")
        var request = URLRequest(url: endpointURL)
        request.httpMethod = "GET"
        request.timeoutInterval = 15

        return try await decodedResponse(for: request)
    }

    private func makeJSONRequest<T: Encodable>(
        baseURL: URL,
        endpoint: String,
        timeout: TimeInterval,
        body: T
    ) throws -> URLRequest {
        let endpointURL = baseURL.appendingPathComponent(endpoint)
        var request = URLRequest(url: endpointURL)
        request.httpMethod = "POST"
        request.timeoutInterval = timeout
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(body)
        return request
    }

    private func decodedResponse<T: Decodable>(for request: URLRequest) async throws -> T {
        do {
            let (data, response) = try await session.data(for: request)
            return try decodeResponseData(data, response: response)
        } catch let error as APIError {
            throw error
        } catch {
            throw APIError.transport(error)
        }
    }

    private func decodedUploadResponse<T: Decodable>(
        for request: URLRequest,
        body: Data
    ) async throws -> T {
        do {
            let (data, response) = try await session.upload(for: request, from: body)
            return try decodeResponseData(data, response: response)
        } catch let error as APIError {
            throw error
        } catch {
            throw APIError.transport(error)
        }
    }

    private func decodeResponseData<T: Decodable>(
        _ data: Data,
        response: URLResponse
    ) throws -> T {
        guard let http = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }
        guard (200...299).contains(http.statusCode) else {
            let message = String(data: data, encoding: .utf8) ?? "Unknown server error"
            throw APIError.server(statusCode: http.statusCode, message: message)
        }

        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            throw APIError.decoding(error)
        }
    }
}

private extension Data {
    mutating func append(_ string: String) {
        if let data = string.data(using: .utf8) {
            append(data)
        }
    }
}
