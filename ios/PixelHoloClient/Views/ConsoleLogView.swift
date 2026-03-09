import SwiftUI

struct ConsoleLogView: View {
    let logs: [ConsoleLogLine]
    var title: String = "Console"

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 6) {
                        ForEach(logs) { line in
                            HStack(alignment: .top, spacing: 8) {
                                Text(Self.timeFormatter.string(from: line.timestamp))
                                    .font(.caption.monospacedDigit())
                                    .foregroundStyle(.secondary)
                                    .frame(width: 84, alignment: .leading)
                                Text(line.text)
                                    .font(.caption.monospaced())
                                    .foregroundStyle(line.isError ? .red : .primary)
                                    .textSelection(.enabled)
                                Spacer(minLength: 0)
                            }
                            .id(line.id)
                        }
                    }
                }
                .onChange(of: logs.count) { _, _ in
                    if let lastID = logs.last?.id {
                        withAnimation(.easeOut(duration: 0.2)) {
                            proxy.scrollTo(lastID, anchor: .bottom)
                        }
                    }
                }
            }
        }
    }

    private static let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        return formatter
    }()
}

#Preview {
    ConsoleLogView(
        logs: [
            ConsoleLogLine(timestamp: .now, text: "Starting preprocess...", isError: false),
            ConsoleLogLine(timestamp: .now, text: "ERROR: missing file", isError: true)
        ]
    )
    .padding()
}

