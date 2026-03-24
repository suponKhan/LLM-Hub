import Foundation
import LlamaCPPRuntime
import RunAnywhere

@MainActor
class LLMBackend: ObservableObject {
    static let shared = LLMBackend()

    @Published var isLoaded: Bool = false
    @Published var currentlyLoadedModel: String? = nil
    @Published var isBackendLoading: Bool = false

    // Generation parameters
    var maxTokens: Int = 2048
    var topK: Int = 64
    var topP: Float = 0.95
    var temperature: Float = 1.0
    var selectedBackend: String = "GPU"
    var enableVision: Bool = true
    var enableAudio: Bool = true
    var enableThinking: Bool = true

    private var isSDKInitialized = false
    private var areModelsRegistered = false

    private init() {}

    private func legacyModelDirectory(for model: AIModel) -> URL? {
        guard let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return nil }
        return documentsDir.appendingPathComponent("models").appendingPathComponent(model.id)
    }

    private func hasAllRequiredFiles(in directory: URL, for model: AIModel) -> Bool {
        guard !model.requiredFileNames.isEmpty else { return false }
        return model.requiredFileNames.allSatisfy { fileName in
            FileManager.default.fileExists(atPath: directory.appendingPathComponent(fileName).path)
        }
    }

    private func migrateLegacyModelIfNeeded(_ model: AIModel) throws -> Bool {
        if RunAnywhere.isModelDownloaded(model.id, framework: model.inferenceFramework) {
            return false
        }

        guard let legacyDir = legacyModelDirectory(for: model),
              FileManager.default.fileExists(atPath: legacyDir.path),
              hasAllRequiredFiles(in: legacyDir, for: model) else {
            return false
        }

        let destinationDir = try SimplifiedFileManager.shared.getModelFolderURL(modelId: model.id, framework: model.inferenceFramework)
        try FileManager.default.createDirectory(at: destinationDir, withIntermediateDirectories: true)

        for fileName in model.requiredFileNames {
            let sourceURL = legacyDir.appendingPathComponent(fileName)
            let destinationURL = destinationDir.appendingPathComponent(fileName)

            if FileManager.default.fileExists(atPath: destinationURL.path) {
                try? FileManager.default.removeItem(at: destinationURL)
            }
            try FileManager.default.copyItem(at: sourceURL, to: destinationURL)
        }

        print("[LLMBackend] migrated legacy model files for \(model.id)")
        return true
    }

    private func filename(from url: URL) -> String {
        URLComponents(url: url, resolvingAgainstBaseURL: false)?.path.split(separator: "/").last.map(String.init) ?? url.lastPathComponent
    }

    private func framework(for model: AIModel) -> InferenceFramework {
        model.inferenceFramework
    }

    private func registerModel(_ model: AIModel) {
        guard let primaryURL = URL(string: model.url) else { return }

        if model.additionalFiles.isEmpty {
            RunAnywhere.registerModel(
                id: model.id,
                name: model.name,
                url: primaryURL,
                framework: framework(for: model),
                modality: model.supportsVision ? .multimodal : .language,
                memoryRequirement: model.sizeBytes,
                supportsThinking: model.supportsThinking
            )
            return
        }

        let descriptors = model.allDownloadURLs.map {
            ModelFileDescriptor(url: $0, filename: filename(from: $0), isRequired: true)
        }

        RunAnywhere.registerMultiFileModel(
            id: model.id,
            name: model.name,
            files: descriptors,
            framework: framework(for: model),
            modality: model.supportsVision ? .multimodal : .language,
            memoryRequirement: model.sizeBytes
        )
    }

    private func ensureSDKReady() async throws {
        if !isSDKInitialized {
            try RunAnywhere.initialize(environment: .development)
            LlamaCPP.register()
            isSDKInitialized = true
        }

        if !areModelsRegistered {
            for model in ModelData.models {
                registerModel(model)
            }
            areModelsRegistered = true
        }

        // Ensure model path APIs are configured before storage checks/migration.
        try await RunAnywhere.completeServicesInitialization()
    }

    func loadModel(_ model: AIModel) async throws {
        isBackendLoading = true
        defer { isBackendLoading = false }

        print("[LLMBackend] loadModel name=\(model.name) visionEnabled=\(enableVision) audioEnabled=\(enableAudio)")

        try await ensureSDKReady()
        _ = try? migrateLegacyModelIfNeeded(model)
        _ = await RunAnywhere.discoverDownloadedModels()

        // Only local load here. Downloads are handled by the model download screen.
        guard RunAnywhere.isModelDownloaded(model.id, framework: model.inferenceFramework) else {
            throw NSError(domain: "LLMBackend", code: -100, userInfo: [NSLocalizedDescriptionKey: "Model is not downloaded locally"])
        }

        try await RunAnywhere.loadModel(model.id)

        isLoaded = true
        currentlyLoadedModel = model.name
    }

    func unloadModel() {
        Task {
            do {
                try await RunAnywhere.unloadModel()
            } catch {
                print("[LLMBackend] unloadModel error=\(error)")
            }
            await MainActor.run {
                self.isLoaded = false
                self.currentlyLoadedModel = nil
            }
        }
    }

    func generate(
        prompt: String,
        imageURL: URL? = nil,
        audioURL: URL? = nil,
        onUpdate: @escaping (String, Int, Double) -> Void
    ) async throws {
        _ = imageURL
        _ = audioURL

        try await ensureSDKReady()

        let options = LLMGenerationOptions(
            maxTokens: max(1, maxTokens),
            temperature: temperature,
            topP: topP,
            streamingEnabled: true
        )

        print("[LLMBackend] generate visionEnabled=\(enableVision) audioEnabled=\(enableAudio) images=0 videos=0")

        let streamResult = try await RunAnywhere.generateStream(prompt, options: options)
        var currentOutput = ""

        for try await token in streamResult.stream {
            currentOutput += token
            onUpdate(currentOutput, 0, 0)
        }

        let result = try await streamResult.result.value
        onUpdate(currentOutput, result.tokensUsed, result.tokensPerSecond)
    }
}
