import AVFoundation
import SwiftUI

struct CameraView: View {
    @StateObject private var camera = CameraManager()

    let profileName: String
    let onVideoReady: (URL) -> Void

    var body: some View {
        VStack(spacing: 16) {
            ZStack(alignment: .topTrailing) {
                CameraPreviewRepresentable(session: camera.session)
                    .aspectRatio(3.0 / 4.0, contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .stroke(Color.white.opacity(0.7), lineWidth: 2)
                    )

                if camera.isRecording {
                    Text("\(camera.remainingSeconds)s")
                        .font(.headline.monospacedDigit())
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(Color.black.opacity(0.7))
                        .foregroundStyle(.white)
                        .clipShape(Capsule())
                        .padding(12)
                }
            }

            if let error = camera.errorMessage {
                Text(error)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            HStack(spacing: 12) {
                Button {
                    if camera.isRecording {
                        camera.stopRecording()
                    } else {
                        camera.startRecording(profileName: profileName, duration: 10)
                    }
                } label: {
                    Text(camera.isRecording ? "Stop Recording" : "Record 10s Video")
                        .fontWeight(.semibold)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!camera.isConfigured)
            }
        }
        .task {
            let granted = await camera.requestPermissions()
            guard granted else { return }
            await camera.configureIfNeeded()
            camera.startSession()
        }
        .onDisappear {
            camera.stopSession()
        }
        .onChange(of: camera.recordedURL) { _, value in
            if let value {
                onVideoReady(value)
            }
        }
    }
}

private struct CameraPreviewRepresentable: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.videoGravity = .resizeAspectFill
        view.previewLayer.session = session
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        if uiView.previewLayer.session !== session {
            uiView.previewLayer.session = session
        }
    }
}

private final class PreviewView: UIView {
    override class var layerClass: AnyClass {
        AVCaptureVideoPreviewLayer.self
    }

    var previewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }
}

