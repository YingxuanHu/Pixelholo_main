# PixelHolo iOS Thin Client

## Suggested Xcode Setup

1. In Xcode, create a new project:
   - iOS App
   - Interface: SwiftUI
   - Language: Swift
2. Add these files to your app target from:
   - `ios/PixelHoloClient/App`
   - `ios/PixelHoloClient/Config`
   - `ios/PixelHoloClient/Media`
   - `ios/PixelHoloClient/Models`
   - `ios/PixelHoloClient/Networking`
   - `ios/PixelHoloClient/ViewModels`
   - `ios/PixelHoloClient/Views`
3. Ensure deployment target supports Swift Concurrency (iOS 15+ recommended).

## Backend URL

Use a reachable backend URL in the app:
- `http://127.0.0.1:8000` for simulator + local backend
- `http://<LAN_IP>:8000` for device testing over local network

## Info.plist

Use `ios/PixelHoloClient/App/Info.plist.template` keys in your app `Info.plist`:
- `NSCameraUsageDescription`
- `NSMicrophoneUsageDescription`
- `NSSpeechRecognitionUsageDescription`
- `NSAppTransportSecurity` (`NSAllowsArbitraryLoads=true` for local HTTP testing)

## Implemented Features

- Phase 1:
  - Server base URL config and persistence.
  - `GET /profiles` integration.
  - Profile list UI split by voice/avatar.
- Phase 2:
  - AVFoundation camera capture.
  - 10s recording flow.
  - center 3:4 export crop.
  - AVAudioRecorder voice capture.
  - multipart upload to `/upload` and `/upload_audio`.
  - create-profile UI.
- Phase 3:
  - delegate-based line streamer (`URLSessionDataDelegate`).
  - `/preprocess` and `/train` streaming log readers.
  - live console log UI.
- Phase 4:
  - NDJSON stream client for `/chat` and `/speak`.
  - background base64 decode for audio and frames.
  - WAV-to-PCM decoder.
  - AVAudioEngine/CADisplayLink avatar sync player.
- Phase 5:
  - speech recognition push-to-talk.
  - backend interrupt (`POST /interrupt`).
  - stop/flush playback path.

## Validation Performed

- iOS SDK compile check (all Swift files) using `swiftc -typecheck` with iOS simulator target.
- Streaming line parser smoke test with a local mock server (chunked lines).
- WAV decoder smoke test with generated WAV payload.
- `/interrupt` API smoke test with a local mock server.

## Device/Simulator Validation Still Needed

- Camera capture, real recording, and final crop framing check in simulator/device UI run.
- Speech recognition UX checks (hold-to-talk timing/permission behavior).
- End-to-end `/chat` and `/speak` with real backend audio+frame sync quality checks.

## ATS Example (dev only)

```xml
<key>NSAppTransportSecurity</key>
<dict>
    <key>NSAllowsArbitraryLoads</key>
    <true/>
</dict>
```
