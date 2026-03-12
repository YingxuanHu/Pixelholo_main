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

        do {
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                throw APIError.invalidResponse
            }
            guard (200...299).contains(http.statusCode) else {
                let message = String(data: data, encoding: .utf8) ?? "Unknown server error"
                throw APIError.server(statusCode: http.statusCode, message: message)
            }

            do {
                let decoded = try decoder.decode(ProfilesResponse.self, from: data)
                return decoded.profiles
            } catch {
                throw APIError.decoding(error)
            }
        } catch let error as APIError {
            throw error
        } catch {
            throw APIError.transport(error)
        }
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

        do {
            let (data, response) = try await session.upload(for: request, from: body)
            guard let http = response as? HTTPURLResponse else {
                throw APIError.invalidResponse
            }
            guard (200...299).contains(http.statusCode) else {
                let message = String(data: data, encoding: .utf8) ?? "Unknown server error"
                throw APIError.server(statusCode: http.statusCode, message: message)
            }

            do {
                return try decoder.decode(UploadResponse.self, from: data)
            } catch {
                throw APIError.decoding(error)
            }
        } catch let error as APIError {
            throw error
        } catch {
            throw APIError.transport(error)
        }
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
        let endpointURL = baseURL.appendingPathComponent("preprocess")
        var urlRequest = URLRequest(url: endpointURL)
        urlRequest.httpMethod = "POST"
        urlRequest.timeoutInterval = 60 * 60
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.httpBody = try JSONEncoder().encode(request)
        return lineStreamer.streamLines(request: urlRequest)
    }

    func startTraining(
        baseURL: URL,
        request: TrainRequest
    ) throws -> AsyncThrowingStream<String, Error> {
        let endpointURL = baseURL.appendingPathComponent("train")
        var urlRequest = URLRequest(url: endpointURL)
        urlRequest.httpMethod = "POST"
        urlRequest.timeoutInterval = 60 * 60
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.httpBody = try JSONEncoder().encode(request)
        return lineStreamer.streamLines(request: urlRequest)
    }

    func interrupt(baseURL: URL) async throws -> InterruptResponse {
        let endpointURL = baseURL.appendingPathComponent("interrupt")
        var request = URLRequest(url: endpointURL)
        request.httpMethod = "POST"
        request.timeoutInterval = 15

        do {
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                throw APIError.invalidResponse
            }
            guard (200...299).contains(http.statusCode) else {
                let message = String(data: data, encoding: .utf8) ?? "Unknown server error"
                throw APIError.server(statusCode: http.statusCode, message: message)
            }
            do {
                return try decoder.decode(InterruptResponse.self, from: data)
            } catch {
                throw APIError.decoding(error)
            }
        } catch let error as APIError {
            throw error
        } catch {
            throw APIError.transport(error)
        }
    }

    func warmup(baseURL: URL, request warmupRequest: WarmupRequest) async throws -> WarmupResponse {
        let endpointURL = baseURL.appendingPathComponent("warmup")
        var request = URLRequest(url: endpointURL)
        request.httpMethod = "POST"
        request.timeoutInterval = 60
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(warmupRequest)

        do {
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                throw APIError.invalidResponse
            }
            guard (200...299).contains(http.statusCode) else {
                let message = String(data: data, encoding: .utf8) ?? "Unknown server error"
                throw APIError.server(statusCode: http.statusCode, message: message)
            }
            do {
                return try decoder.decode(WarmupResponse.self, from: data)
            } catch {
                throw APIError.decoding(error)
            }
        } catch let error as APIError {
            throw error
        } catch {
            throw APIError.transport(error)
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
