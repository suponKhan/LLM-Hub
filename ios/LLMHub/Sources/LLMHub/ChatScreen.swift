import Foundation
import PhotosUI
import RunAnywhere
import SwiftUI
import UniformTypeIdentifiers

// MARK: - Chat ViewModel
@MainActor
class ChatViewModel: ObservableObject {
    @Published var inputText: String = ""
    @Published var isGenerating: Bool = false
    @Published var tokensPerSecond: Double = 0
    @Published var totalTokens: Int = 0
    @Published var selectedModelName: String = AppSettings.shared.localized("no_model_selected")
    @Published var isBackendLoading: Bool = false
    
    // Config Properties (Persisted)
    @AppStorage("chat_max_tokens") var maxTokens: Double = 512
    @AppStorage("chat_top_k") var topK: Double = 64
    @AppStorage("chat_top_p") var topP: Double = 0.95
    @AppStorage("chat_temperature") var temperature: Double = 1.0
    @AppStorage("chat_selected_backend") var selectedBackend: String = "GPU"
    @AppStorage("chat_enable_vision") var enableVision: Bool = true
    @AppStorage("chat_enable_audio") var enableAudio: Bool = true
    @AppStorage("chat_enable_thinking") var enableThinking: Bool = true

    private let chatStore = ChatStore.shared
    private let llmBackend = LLMBackend.shared
    @Published var currentSessionId: UUID = UUID()
    private var activeGeneratingMessageId: UUID?
    
    // Compute current title from sessionId
    var currentTitle: String {
        get { chatStore.chatSessions.first(where: { $0.id == currentSessionId })?.title ?? "" }
        set {
            if let index = chatStore.chatSessions.firstIndex(where: { $0.id == currentSessionId }) {
                chatStore.chatSessions[index].title = newValue
                chatStore.saveSessions()
            }
        }
    }

    var chatSessions: [ChatSession] { chatStore.chatSessions }

    var latestUserMessageId: UUID? {
        messages.last(where: { $0.isFromUser })?.id
    }

    var latestAssistantMessageId: UUID? {
        messages.last(where: { !$0.isFromUser && !$0.isGenerating && !$0.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty })?.id
    }
    
    var messages: [ChatMessage] {
        get {
            chatStore.chatSessions.first(where: { $0.id == currentSessionId })?.messages ?? []
        }
        set {
            if let index = chatStore.chatSessions.firstIndex(where: { $0.id == currentSessionId }) {
                chatStore.chatSessions[index].messages = newValue
                chatStore.saveSessions()
                objectWillChange.send()
            }
        }
    }

    init() {
        do {
            try RunAnywhere.initialize(environment: .development)
        } catch {
            // Ignore repeated initialization attempts.
        }

        Task {
            _ = await RunAnywhere.discoverDownloadedModels()
        }

        if let empty = chatStore.chatSessions.first(where: { $0.messages.isEmpty }) {
            currentSessionId = empty.id
        } else {
            newChat()
        }
    }

    var loadedModelName: String? { llmBackend.currentlyLoadedModel }

    func loadModelIfNecessary(force: Bool = false) async {
        syncBackendSettings()

        guard selectedModelName != AppSettings.shared.localized("no_model_selected") else { return }
        if !force && llmBackend.currentlyLoadedModel == selectedModelName { return }
        
        guard let model = ModelData.models.first(where: { $0.name == selectedModelName }) else { return }
        
        isBackendLoading = true
        defer { isBackendLoading = false }
        
        do {
            try await llmBackend.loadModel(model)
        } catch {
            print("Failed to load model: \(error)")
        }
    }

    private func syncBackendSettings() {
        llmBackend.maxTokens = Int(maxTokens)
        llmBackend.topK = Int(topK)
        llmBackend.topP = Float(topP)
        llmBackend.temperature = Float(temperature)
        llmBackend.enableVision = enableVision
        llmBackend.enableAudio = enableAudio
        llmBackend.enableThinking = enableThinking
        llmBackend.selectedBackend = selectedBackend
    }

    func unloadModel() {
        llmBackend.isLoaded = false
        llmBackend.currentlyLoadedModel = nil
    }

    @discardableResult
    func sendMessage(imageURL: URL? = nil, audioURL: URL? = nil) -> Bool {
        let selectedModel = ModelData.models.first(where: { $0.name == selectedModelName })
        let effectiveImageURL = (enableVision && selectedModel?.supportsVision == true) ? imageURL : nil
        let effectiveAudioURL = (enableAudio && selectedModel?.supportsAudio == true) ? audioURL : nil

        let input = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        let hasAttachment = effectiveImageURL != nil || effectiveAudioURL != nil
        guard !input.isEmpty || hasAttachment else { return false }
        guard !isGenerating else { return false }

        let userText = input.isEmpty ? "[Attachment]" : input
        let userMsg = ChatMessage(content: userText, isFromUser: true)
        messages.append(userMsg)
        inputText = ""

        // Auto-update title if it's "New Chat"
        if currentTitle == AppSettings.shared.localized("drawer_new_chat") {
            currentTitle = String(userText.prefix(20))
        }

        let aiMsg = ChatMessage(content: "", isFromUser: false, isGenerating: true)
        messages.append(aiMsg)
        activeGeneratingMessageId = aiMsg.id
        isGenerating = true

        streamingTask = Task {
            await loadModelIfNecessary()
            
            do {
                if !llmBackend.isLoaded {
                    // Fail if still not loaded
                    let msg = AppSettings.shared.localized("please_download_model")
                    await updateLastAIMessage(content: msg, isGenerating: false)
                    await MainActor.run { self.isGenerating = false }
                    return
                }
                
                try await llmBackend.generate(prompt: userText, imageURL: effectiveImageURL, audioURL: effectiveAudioURL) { [weak self] content, tokens, tps in
                    Task { @MainActor [weak self] in
                        guard let self = self else { return }
                        self.updateLastAIMessageSync(content: content, tokens: tokens, tps: tps)
                    }
                }
                await MainActor.run { self.finishGeneratingMessage() }
            } catch {
                await updateLastAIMessage(content: "Error: \(error.localizedDescription)", isGenerating: false)
            }
            
            await MainActor.run {
                self.isGenerating = false
                self.activeGeneratingMessageId = nil
            }
        }

        return true
    }

    private func updateLastAIMessage(content: String, isGenerating: Bool) async {
        await MainActor.run {
            updateLastAIMessageSync(content: content, isGenerating: isGenerating)
        }
    }

    private func updateLastAIMessageSync(content: String, tokens: Int = 0, tps: Double = 0, isGenerating: Bool = true) {
        let targetIndex: Int?
        if let activeId = activeGeneratingMessageId {
            targetIndex = messages.firstIndex(where: { $0.id == activeId })
        } else {
            targetIndex = messages.indices.last
        }

        if let idx = targetIndex, !messages[idx].isFromUser {
            var msgs = self.messages
            msgs[idx].content = content
            msgs[idx].isGenerating = isGenerating
            self.totalTokens = tokens
            self.tokensPerSecond = tps
            msgs[idx].tokenCount = tokens > 0 ? tokens : msgs[idx].tokenCount
            msgs[idx].tokensPerSecond = tps > 0 ? tps : msgs[idx].tokensPerSecond
            self.messages = msgs
        }
    }

    private func finishGeneratingMessage() {
        let targetIndex: Int?
        if let activeId = activeGeneratingMessageId {
            targetIndex = messages.firstIndex(where: { $0.id == activeId })
        } else {
            targetIndex = messages.indices.last
        }

        if let idx = targetIndex, !messages[idx].isFromUser {
            var msgs = self.messages
            msgs[idx].isGenerating = false
            if totalTokens > 0 {
                msgs[idx].tokenCount = totalTokens
                msgs[idx].tokensPerSecond = tokensPerSecond
            }
            self.messages = msgs
        }
        activeGeneratingMessageId = nil
    }

    func stopGeneration() {
        streamingTask?.cancel()
        streamingTask = nil
        if let activeId = activeGeneratingMessageId,
           let idx = messages.firstIndex(where: { $0.id == activeId }),
           !messages[idx].isFromUser {
            messages[idx].isGenerating = false
        } else if let idx = messages.indices.last, !messages[idx].isFromUser {
            messages[idx].isGenerating = false
        }
        activeGeneratingMessageId = nil
        isGenerating = false
    }

    func copyMessage(_ message: ChatMessage) {
        UIPasteboard.general.string = message.content
    }

    func newChat() {
        let session = ChatSession(title: AppSettings.shared.localized("drawer_new_chat"))
        chatStore.addSession(session)
        currentSessionId = session.id
        objectWillChange.send()
    }

    func deleteSession(_ id: UUID) {
        chatStore.deleteSession(id: id)
        if currentSessionId == id {
            if let first = chatSessions.first {
                currentSessionId = first.id
            } else {
                newChat()
            }
        }
        objectWillChange.send()
    }

    func regenerateResponse(for assistantMessageId: UUID) {
        guard !isGenerating else { return }
        guard let assistantIndex = messages.firstIndex(where: { $0.id == assistantMessageId && !$0.isFromUser }) else { return }
        guard let userIndex = messages[..<assistantIndex].lastIndex(where: { $0.isFromUser }) else { return }

        let prompt = messages[userIndex].content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else { return }

        var msgs = messages
        msgs[assistantIndex].content = ""
        msgs[assistantIndex].isGenerating = true
        msgs[assistantIndex].tokenCount = nil
        msgs[assistantIndex].tokensPerSecond = nil
        messages = msgs

        totalTokens = 0
        tokensPerSecond = 0
        activeGeneratingMessageId = assistantMessageId
        isGenerating = true

        streamingTask = Task {
            await loadModelIfNecessary()

            do {
                if !llmBackend.isLoaded {
                    let msg = AppSettings.shared.localized("please_download_model")
                    await updateLastAIMessage(content: msg, isGenerating: false)
                    await MainActor.run {
                        self.isGenerating = false
                        self.activeGeneratingMessageId = nil
                    }
                    return
                }

                try await llmBackend.generate(prompt: prompt) { [weak self] content, tokens, tps in
                    Task { @MainActor [weak self] in
                        guard let self = self else { return }
                        self.updateLastAIMessageSync(content: content, tokens: tokens, tps: tps)
                    }
                }
                await MainActor.run { self.finishGeneratingMessage() }
            } catch {
                await updateLastAIMessage(content: "Error: \(error.localizedDescription)", isGenerating: false)
            }

            await MainActor.run {
                self.isGenerating = false
                self.activeGeneratingMessageId = nil
            }
        }
    }

    func editAssistantMessage(_ messageId: UUID, newText: String) {
        let trimmed = newText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        guard let idx = messages.firstIndex(where: { $0.id == messageId && !$0.isFromUser }) else { return }
        var msgs = messages
        msgs[idx].content = trimmed
        messages = msgs
    }

    func editUserPrompt(_ messageId: UUID, newText: String) {
        guard !isGenerating else { return }
        let trimmed = newText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        guard let userIndex = messages.firstIndex(where: { $0.id == messageId && $0.isFromUser }) else { return }

        var msgs = messages
        msgs[userIndex].content = trimmed
        messages = msgs

        let nextIndex = userIndex + 1
        if nextIndex < messages.count,
           let assistantIndex = messages[nextIndex...].firstIndex(where: { !$0.isFromUser }) {
            regenerateResponse(for: messages[assistantIndex].id)
        }
    }

    private var streamingTask: Task<Void, Never>?
}

// MARK: - Message Bubble
struct MessageBubble: View {
    @EnvironmentObject var settings: AppSettings
    let message: ChatMessage
    let onCopy: () -> Void
    let onEditUserMessage: ((String) -> Void)?
    let onEditAssistantMessage: ((String) -> Void)?
    let onRegenerateResponse: (() -> Void)?
    @State private var showActions = false
    @State private var isEditing = false
    @State private var editedText = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if message.isFromUser {
                HStack {
                    Spacer(minLength: 40)
                    if isEditing {
                        VStack(alignment: .trailing, spacing: 8) {
                            TextEditor(text: $editedText)
                                .frame(minHeight: 90)
                                .padding(8)
                                .background(Color(.secondarySystemBackground))
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                            HStack(spacing: 8) {
                                Button {
                                    isEditing = false
                                    editedText = ""
                                } label: {
                                    Image(systemName: "xmark")
                                }
                                Button {
                                    let trimmed = editedText.trimmingCharacters(in: .whitespacesAndNewlines)
                                    if !trimmed.isEmpty {
                                        onEditUserMessage?(trimmed)
                                        isEditing = false
                                        editedText = ""
                                    }
                                } label: {
                                    Image(systemName: "checkmark")
                                }
                                .disabled(editedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                            }
                            .font(.caption)
                            .foregroundColor(.secondary)
                        }
                        .frame(maxWidth: 320)
                    } else {
                        Text(message.content)
                            .font(.body)
                            .foregroundColor(.white)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)
                            .background(
                                RoundedRectangle(cornerRadius: 18)
                                    .fill(LinearGradient(colors: [Color.indigo, Color.purple], startPoint: .topLeading, endPoint: .bottomTrailing))
                            )
                            .onLongPressGesture {
                                showActions = true
                            }
                    }
                }
            } else {
                if message.isGenerating && message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    TypingIndicator()
                        .padding(.vertical, 6)
                } else {
                    if isEditing {
                        VStack(alignment: .leading, spacing: 8) {
                            TextEditor(text: $editedText)
                                .frame(minHeight: 100)
                                .padding(8)
                                .background(Color(.secondarySystemBackground))
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                            HStack(spacing: 10) {
                                Button {
                                    isEditing = false
                                    editedText = ""
                                } label: {
                                    Image(systemName: "xmark")
                                }
                                Button {
                                    let trimmed = editedText.trimmingCharacters(in: .whitespacesAndNewlines)
                                    if !trimmed.isEmpty {
                                        onEditAssistantMessage?(trimmed)
                                        isEditing = false
                                        editedText = ""
                                    }
                                } label: {
                                    Image(systemName: "checkmark")
                                }
                                .disabled(editedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                            }
                            .font(.caption)
                            .foregroundColor(.secondary)
                        }
                    } else {
                        RenderMessageSegments(displayContent: message.content)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .onLongPressGesture {
                                showActions = true
                            }
                    }
                }
            }

            if !isEditing && !message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                HStack(spacing: 8) {
                    if message.isFromUser {
                        Spacer()
                    }

                    Button(action: onCopy) {
                        Image(systemName: "doc.on.doc")
                    }
                    .buttonStyle(.plain)
                    .foregroundColor(.secondary)

                    if message.isFromUser, let onEditUserMessage {
                        Button {
                            editedText = message.content
                            isEditing = true
                        } label: {
                            Image(systemName: "pencil")
                        }
                        .buttonStyle(.plain)
                        .foregroundColor(.secondary)
                    }

                    if !message.isFromUser, let onEditAssistantMessage {
                        Button {
                            editedText = message.content
                            isEditing = true
                        } label: {
                            Image(systemName: "pencil")
                        }
                        .buttonStyle(.plain)
                        .foregroundColor(.secondary)
                    }

                    if !message.isFromUser, let onRegenerateResponse {
                        Button(action: onRegenerateResponse) {
                            Image(systemName: "arrow.clockwise")
                        }
                        .buttonStyle(.plain)
                        .foregroundColor(.secondary)
                    }

                    if !message.isFromUser,
                       let tokenCount = message.tokenCount,
                       let tps = message.tokensPerSecond,
                       tokenCount > 0 {
                        Spacer()
                        Label(String(format: settings.localized("tokens_per_second_format"), tokenCount, tps), systemImage: "bolt.fill")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
            }

            Text(message.timestamp, style: .time)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .confirmationDialog(settings.localized("more_options"), isPresented: $showActions) {
            Button(settings.localized("copy_message")) {
                onCopy()
            }
            Button(settings.localized("cancel"), role: .cancel) {}
        }
    }
}

// MARK: - Typing Indicator
struct TypingIndicator: View {
    @State private var phase = 0.0

    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<3) { i in
                Circle()
                    .fill(Color.secondary)
                    .frame(width: 6, height: 6)
                    .scaleEffect(1.0 + 0.4 * sin(phase + Double(i) * .pi / 1.5))
            }
        }
        .onAppear {
            withAnimation(.linear(duration: 1).repeatForever(autoreverses: false)) {
                phase = .pi * 2
            }
        }
    }
}

private enum ParsedSegment {
    case text(String)
    case code(language: String?, content: String)
}

private struct RenderMessageSegments: View {
    let displayContent: String

    var body: some View {
        let segments = parseSegments(normalized(displayContent))
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(segments.enumerated()), id: \.offset) { item in
                let segment = item.element
                switch segment {
                case .text(let text):
                    MarkdownMessageText(text: text)
                case .code(let language, let content):
                    VStack(alignment: .leading, spacing: 6) {
                        if let language, !language.isEmpty {
                            Text(language)
                                .font(.caption2.weight(.semibold))
                                .foregroundColor(.secondary)
                        }
                        ScrollView(.horizontal, showsIndicators: false) {
                            Text(content.trimmingCharacters(in: .newlines))
                                .font(.system(.body, design: .monospaced))
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    .padding(10)
                    .background(Color.secondary.opacity(0.12))
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                }
            }
        }
    }

    private func normalized(_ raw: String) -> String {
        var value = raw
        let markers = ["<end_of_turn>", "<|eot_id|>", "<|endoftext|>", "</s>"]
        for marker in markers {
            value = value.replacingOccurrences(of: marker, with: "")
        }

        // Render block math in a code-style block for readable display.
        value = value.replacingOccurrences(
            of: #"\$\$([\s\S]*?)\$\$"#,
            with: "```math\n$1\n```",
            options: .regularExpression
        )
        // Render inline math as inline code-style segment.
        value = value.replacingOccurrences(
            of: #"\$(?!\$)([^\n$]+)\$"#,
            with: "`$1`",
            options: .regularExpression
        )
        return value
    }

    private func parseSegments(_ input: String) -> [ParsedSegment] {
        let pattern = #"```([a-zA-Z0-9_+-]*)\n([\s\S]*?)```"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
            return [.text(input)]
        }

        let nsInput = input as NSString
        let matches = regex.matches(in: input, options: [], range: NSRange(location: 0, length: nsInput.length))
        if matches.isEmpty {
            return [.text(input)]
        }

        var segments: [ParsedSegment] = []
        var cursor = 0

        for match in matches {
            if match.range.location > cursor {
                let textPart = nsInput.substring(with: NSRange(location: cursor, length: match.range.location - cursor))
                if !textPart.isEmpty {
                    segments.append(.text(textPart))
                }
            }

            let language: String? = {
                let langRange = match.range(at: 1)
                guard langRange.location != NSNotFound else { return nil }
                let lang = nsInput.substring(with: langRange).trimmingCharacters(in: .whitespacesAndNewlines)
                return lang.isEmpty ? nil : lang
            }()

            let code = nsInput.substring(with: match.range(at: 2))
            segments.append(.code(language: language, content: code))
            cursor = match.range.location + match.range.length
        }

        if cursor < nsInput.length {
            let trailing = nsInput.substring(from: cursor)
            if !trailing.isEmpty {
                segments.append(.text(trailing))
            }
        }

        return segments
    }
}

private struct MarkdownMessageText: View {
    let text: String

    var body: some View {
        let normalizedText = text.replacingOccurrences(of: "\\n", with: "\n")
        let lines = normalizedText.components(separatedBy: "\n")

        VStack(alignment: .leading, spacing: 4) {
            ForEach(Array(lines.enumerated()), id: \.offset) { indexedLine in
                let line = indexedLine.element
                if line.isEmpty {
                    Color.clear
                        .frame(height: 10)
                } else if let attributed = try? AttributedString(
                    markdown: line,
                    options: .init(
                        interpretedSyntax: .full,
                        failurePolicy: .returnPartiallyParsedIfPossible
                    )
                ) {
                    Text(attributed)
                        .font(.body)
                        .lineSpacing(4)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                } else {
                    Text(line)
                        .font(.body)
                        .lineSpacing(4)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
            }
        }
    }
}

// MARK: - Drawer Panel
struct ChatDrawerPanel: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject var vm: ChatViewModel
    let onClose: () -> Void
    let onNavigateBack: () -> Void
    let onNavigateToModels: () -> Void
    let onNavigateToSettings: () -> Void
    @State private var showDeleteAllAlert = false

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Button {
                        vm.newChat()
                        onClose()
                    } label: {
                        Label(settings.localized("drawer_new_chat"), systemImage: "plus.bubble.fill")
                            .foregroundColor(.indigo)
                            .fontWeight(.semibold)
                    }
                }

                Section(settings.localized("drawer_recent_chats")) {
                    if vm.chatSessions.isEmpty {
                        Text(settings.localized("drawer_no_chats"))
                            .foregroundColor(.secondary)
                            .font(.subheadline)
                    } else {
                        ForEach(vm.chatSessions) { session in
                            Button {
                                vm.currentSessionId = session.id
                                onClose()
                            } label: {
                                HStack {
                                    Image(systemName: "bubble.left.fill")
                                        .foregroundColor(.indigo.opacity(0.7))
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(session.title)
                                            .foregroundColor(.primary)
                                            .lineLimit(1)
                                        Text(session.createdAt, style: .date)
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    }
                                    Spacer()
                                    if session.id == vm.currentSessionId {
                                        Image(systemName: "checkmark.circle.fill")
                                            .foregroundColor(.indigo)
                                    }
                                }
                            }
                            .swipeActions(edge: .trailing) {
                                Button(role: .destructive) {
                                    vm.deleteSession(session.id)
                                } label: {
                                    Label(settings.localized("action_delete"), systemImage: "trash")
                                }
                            }
                        }
                    }
                }

                Section {
                    Button {
                        onClose()
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                            onNavigateToModels()
                        }
                    } label: {
                        Label(settings.localized("drawer_download_models"), systemImage: "square.and.arrow.down")
                    }
                    Button {
                        onClose()
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                            onNavigateToSettings()
                        }
                    } label: {
                        Label(settings.localized("drawer_settings"), systemImage: "gearshape")
                    }
                    if !vm.chatSessions.isEmpty {
                        Button(role: .destructive) {
                            showDeleteAllAlert = true
                        } label: {
                            Label(settings.localized("drawer_clear_all_chats"), systemImage: "trash")
                        }
                    }
                }
            }
            .navigationTitle(settings.localized("drawer_title"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    // Back arrow to Home - same as Android drawer's ArrowBack
                    Button {
                        onClose()
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                            onNavigateBack()
                        }
                    } label: {
                        Image(systemName: "arrow.left")
                    }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button(settings.localized("done"), action: onClose)
                }
            }
        }
        .alert(settings.localized("dialog_delete_all_chats_title"), isPresented: $showDeleteAllAlert) {
            Button(settings.localized("action_delete_all"), role: .destructive) {
                ChatStore.shared.clearAll()
                vm.newChat()
            }
            Button(settings.localized("action_cancel"), role: .cancel) {}
        } message: {
            Text(settings.localized("dialog_delete_all_chats_message"))
        }
    }
}


// MARK: - ChatScreen
struct ChatScreen: View {
    @EnvironmentObject var settings: AppSettings
    @StateObject private var vm = ChatViewModel()
    var onNavigateToSettings: () -> Void
    var onNavigateToModels: () -> Void
    var onNavigateBack: () -> Void

    @State private var showDrawer = false
    @State private var showSettings = false
    @State private var copiedMessageId: UUID? = nil
    @State private var selectedImageItem: PhotosPickerItem?
    @State private var attachedImageURL: URL?
    @State private var attachedAudioURL: URL?
    @State private var showAudioImporter = false
    @FocusState private var isComposerFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            if !vm.messages.isEmpty {
                HStack(spacing: 12) {
                    Button {
                        showSettings = true
                    } label: {
                        HStack(spacing: 4) {
                            Text(vm.selectedModelName)
                                .font(.caption.bold())
                            Image(systemName: "chevron.down")
                                .font(.system(size: 8, weight: .bold))
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                        .background(vm.isBackendLoading ? .orange.opacity(0.2) : .indigo.opacity(0.1))
                        .foregroundColor(vm.isBackendLoading ? .orange : .indigo)
                        .clipShape(Capsule())
                    }
                    Spacer()
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(.thinMaterial)
            }

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 12) {
                        if vm.messages.isEmpty {
                            emptyState
                        } else {
                            ForEach(vm.messages) { msg in
                                let isLatestAssistant = (msg.id == vm.latestAssistantMessageId)
                                let canRegenerate = isLatestAssistant && !vm.isGenerating && !msg.isGenerating
                                let canEditUser = msg.isFromUser && msg.id == vm.latestUserMessageId && !vm.isGenerating
                                let canEditAssistant = !msg.isFromUser && !vm.isGenerating && !msg.isGenerating
                                let regenerateAction: (() -> Void)? = canRegenerate ? {
                                    vm.regenerateResponse(for: msg.id)
                                } : nil
                                MessageBubble(
                                    message: msg,
                                    onCopy: {
                                        vm.copyMessage(msg)
                                        copiedMessageId = msg.id
                                        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                                            copiedMessageId = nil
                                        }
                                    },
                                    onEditUserMessage: { updatedPrompt in
                                        if canEditUser {
                                            vm.editUserPrompt(msg.id, newText: updatedPrompt)
                                        }
                                    },
                                    onEditAssistantMessage: { updatedResponse in
                                        if canEditAssistant {
                                            vm.editAssistantMessage(msg.id, newText: updatedResponse)
                                        }
                                    },
                                    onRegenerateResponse: regenerateAction
                                )
                                .id(msg.id)
                                .padding(.horizontal, 16)
                            }
                        }
                    }
                    .padding(.vertical, 12)
                }
                .scrollDismissesKeyboard(.interactively)
                .onTapGesture {
                    isComposerFocused = false
                }
                .onChange(of: vm.messages.count) { _, _ in
                    if let last = vm.messages.last {
                        withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                    }
                }
                .onChange(of: vm.currentSessionId) { _, _ in
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                        if let last = vm.messages.last {
                            withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                        }
                    }
                }
                .onChange(of: vm.messages.last?.content ?? "") { _, _ in
                    if vm.isGenerating, let last = vm.messages.last {
                        withAnimation(.easeOut(duration: 0.12)) {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
                .onChange(of: isComposerFocused) { _, focused in
                    if focused, let last = vm.messages.last {
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                            withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                        }
                    }
                }
            }

            if let _ = copiedMessageId {
                Text(settings.localized("message_copied"))
                    .font(.caption)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(.ultraThinMaterial)
                    .clipShape(Capsule())
                    .transition(.scale.combined(with: .opacity))
            }

            Divider()

            if attachedImageURL != nil || attachedAudioURL != nil {
                HStack(spacing: 8) {
                    if attachedImageURL != nil {
                        attachmentPill(label: settings.localized("vision"), icon: "photo") {
                            attachedImageURL = nil
                            selectedImageItem = nil
                        }
                    }
                    if attachedAudioURL != nil {
                        attachmentPill(label: settings.localized("audio"), icon: "waveform") {
                            attachedAudioURL = nil
                        }
                    }
                    Spacer()
                }
                .padding(.horizontal, 12)
                .padding(.top, 6)
            }

            HStack(spacing: 10) {
                let selectedModel = ModelData.models.first(where: { $0.name == vm.selectedModelName })
                let canAttachVision = (selectedModel?.supportsVision == true) && vm.enableVision
                let canAttachAudio = (selectedModel?.supportsAudio == true) && vm.enableAudio

                if canAttachVision {
                    PhotosPicker(selection: $selectedImageItem, matching: .images) {
                        Image(systemName: "photo")
                            .font(.system(size: 18, weight: .semibold))
                            .foregroundColor(.indigo)
                    }
                    .disabled(vm.isGenerating)
                }

                if canAttachAudio {
                    Button {
                        showAudioImporter = true
                    } label: {
                        Image(systemName: "waveform")
                            .font(.system(size: 18, weight: .semibold))
                            .foregroundColor(.indigo)
                    }
                    .disabled(vm.isGenerating)
                }

                TextField(settings.localized("type_a_message"), text: $vm.inputText, axis: .vertical)
                    .lineLimit(1...5)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(Color(.secondarySystemBackground))
                    .clipShape(RoundedRectangle(cornerRadius: 22))
                    .focused($isComposerFocused)
                    .onSubmit {
                        if vm.sendMessage(imageURL: attachedImageURL, audioURL: attachedAudioURL) {
                            attachedImageURL = nil
                            attachedAudioURL = nil
                            selectedImageItem = nil
                        }
                    }

                Button {
                    isComposerFocused = false
                    if vm.isGenerating {
                        vm.stopGeneration()
                    } else {
                        if vm.sendMessage(imageURL: attachedImageURL, audioURL: attachedAudioURL) {
                            attachedImageURL = nil
                            attachedAudioURL = nil
                            selectedImageItem = nil
                        }
                    }
                } label: {
                    Image(systemName: vm.isGenerating ? "stop.circle.fill" : "arrow.up.circle.fill")
                        .font(.system(size: 34))
                        .foregroundStyle(vm.isGenerating ? .red : .indigo)
                }
                .disabled(
                    !vm.isGenerating
                        && vm.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                        && attachedImageURL == nil
                        && attachedAudioURL == nil
                )
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(.background)
        }
        .navigationTitle(vm.chatSessions.first(where: { $0.id == vm.currentSessionId })?.title ?? settings.localized("chat"))
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button {
                    showDrawer = true
                } label: {
                    Image(systemName: "line.3.horizontal")
                }
            }
            ToolbarItem(placement: .navigationBarTrailing) {
                Button {
                    showSettings = true
                } label: {
                    Image(systemName: "slider.horizontal.3")
                }
            }
        }
        .sheet(isPresented: $showSettings) {
             ChatSettingsSheet(vm: vm)
        }
        .sheet(isPresented: $showDrawer) {
            ChatDrawerPanel(
                vm: vm,
                onClose: { showDrawer = false },
                onNavigateBack: onNavigateBack,
                onNavigateToModels: onNavigateToModels,
                onNavigateToSettings: onNavigateToSettings
            )
        }
        .fileImporter(
            isPresented: $showAudioImporter,
            allowedContentTypes: [.audio, .mpeg4Audio],
            allowsMultipleSelection: false
        ) { result in
            guard case .success(let urls) = result, let sourceURL = urls.first else { return }
            attachedAudioURL = copyAttachmentToTemp(sourceURL, preferredExtension: sourceURL.pathExtension.isEmpty ? "m4a" : sourceURL.pathExtension)
        }
        .onChange(of: selectedImageItem) { _, item in
            guard let item else {
                attachedImageURL = nil
                return
            }

            Task {
                if let data = try? await item.loadTransferable(type: Data.self) {
                    await MainActor.run {
                        attachedImageURL = writeAttachmentData(data, preferredExtension: "jpg")
                    }
                }
            }
        }
        .onChange(of: vm.enableVision) { _, enabled in
            if !enabled {
                attachedImageURL = nil
                selectedImageItem = nil
            }
        }
        .onChange(of: vm.enableAudio) { _, enabled in
            if !enabled {
                attachedAudioURL = nil
            }
        }
        .onChange(of: vm.selectedModelName) { _, _ in
            let selectedModel = ModelData.models.first(where: { $0.name == vm.selectedModelName })
            let canAttachVision = (selectedModel?.supportsVision == true) && vm.enableVision
            let canAttachAudio = (selectedModel?.supportsAudio == true) && vm.enableAudio

            if !canAttachVision {
                attachedImageURL = nil
                selectedImageItem = nil
            }
            if !canAttachAudio {
                attachedAudioURL = nil
            }
        }
    }

    private func attachmentPill(label: String, icon: String, onRemove: @escaping () -> Void) -> some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
            Text(label)
            Button(action: onRemove) {
                Image(systemName: "xmark.circle.fill")
            }
        }
        .font(.caption)
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(Color(.secondarySystemBackground))
        .clipShape(Capsule())
    }

    private func writeAttachmentData(_ data: Data, preferredExtension: String) -> URL? {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("llmhub_attachments", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let ext = preferredExtension.isEmpty ? "bin" : preferredExtension
        let url = dir.appendingPathComponent(UUID().uuidString).appendingPathExtension(ext)
        do {
            try data.write(to: url, options: .atomic)
            return url
        } catch {
            return nil
        }
    }

    private func copyAttachmentToTemp(_ sourceURL: URL, preferredExtension: String) -> URL? {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("llmhub_attachments", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let ext = preferredExtension.isEmpty ? sourceURL.pathExtension : preferredExtension
        let destinationURL = dir.appendingPathComponent(UUID().uuidString).appendingPathExtension(ext)
        do {
            if FileManager.default.fileExists(atPath: destinationURL.path) {
                try FileManager.default.removeItem(at: destinationURL)
            }
            try FileManager.default.copyItem(at: sourceURL, to: destinationURL)
            return destinationURL
        } catch {
            return nil
        }
    }

    var emptyState: some View {
        VStack(spacing: 20) {
            Spacer(minLength: 60)
            if let uiImage = UIImage(named: "Icon") {
                Image(uiImage: uiImage)
                    .resizable()
                    .scaledToFit()
                    .frame(width: 80, height: 80)
                    .cornerRadius(16)
            } else {
                Image(systemName: "cpu")
                    .font(.system(size: 64))
                    .foregroundStyle(.linearGradient(colors: [.indigo, .purple], startPoint: .top, endPoint: .bottom))
            }
            
            Text(settings.localized("welcome_to_llm_hub"))
                .font(.title2.bold())
                
            if downloadedModels.isEmpty {
                Text(settings.localized("no_models_downloaded"))
                    .foregroundColor(.secondary)
                Button {
                    onNavigateToModels()
                } label: {
                    Label(settings.localized("download_a_model"), systemImage: "arrow.down.circle")
                }
                .buttonStyle(.borderedProminent)
            } else if vm.selectedModelName == settings.localized("no_model_selected") {
                Text(settings.localized("load_model_to_start"))
                    .foregroundColor(.secondary)
            } else {
                Text(vm.selectedModelName)
                    .font(.caption)
                    .padding(.horizontal, 12).padding(.vertical, 6)
                    .background(Color.secondary.opacity(0.2))
                    .clipShape(Capsule())
                Text(settings.localized("start_chatting"))
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        .padding(.horizontal, 32)
    }
    
    private var downloadedModels: [AIModel] {
        let legacyModelsDir: URL? = {
            guard let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return nil }
            return documentsDir.appendingPathComponent("models")
        }()

        return ModelData.models.filter { model in
            if RunAnywhere.isModelDownloaded(model.id, framework: model.inferenceFramework) {
                return true
            }

            guard let legacyModelsDir else { return false }
            let legacyModelDir = legacyModelsDir.appendingPathComponent(model.id)
            guard FileManager.default.fileExists(atPath: legacyModelDir.path) else { return false }
            guard !model.requiredFileNames.isEmpty else { return false }

            return model.requiredFileNames.allSatisfy { fileName in
                let fileURL = legacyModelDir.appendingPathComponent(fileName)
                return FileManager.default.fileExists(atPath: fileURL.path)
            }
        }
    }
}
