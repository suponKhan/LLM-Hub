import Foundation
import MLX
import MLXNN
import MLXRandom
import MLXLLM
import MLXLMCommon
import Hub
import Tokenizers

@MainActor
class LLMBackend: ObservableObject {
    static let shared = LLMBackend()
    
    @Published var isLoaded: Bool = false
    @Published var currentlyLoadedModel: String? = nil
    @Published var isBackendLoading: Bool = false
    
    // Generation Parameters
    var maxTokens: Int = 2048
    var topK: Int = 64
    var topP: Float = 0.95
    var temperature: Float = 1.0
    var selectedBackend: String = "GPU"
    var enableVision: Bool = true
    var enableAudio: Bool = true
    var enableThinking: Bool = true
    
    // MLX Model Container
    private var modelContainer: ModelContainer?
    private var currentContextWindowSize: Int = 2048

    private let stopMarkers: [String] = [
        "<end_of_turn>",
        "<|eot_id|>",
        "<|endoftext|>",
        "</s>",
        "<eos>",
        "<|im_end|>"
    ]
    
    private init() {}

    private func intValue(from any: Any?) -> Int? {
        switch any {
        case let value as Int:
            return value
        case let value as Int64:
            guard value <= Int64(Int.max), value >= Int64(Int.min) else { return nil }
            return Int(value)
        case let value as Double:
            guard value.isFinite,
                value <= Double(Int.max),
                value >= Double(Int.min)
            else {
                return nil
            }
            return Int(value)
        case let value as NSNumber:
            let number = value.doubleValue
            guard number.isFinite,
                number <= Double(Int.max),
                number >= Double(Int.min)
            else {
                return nil
            }
            return Int(number)
        case let value as String:
            if let intValue = Int(value) {
                return intValue
            }
            if let number = Double(value),
                number.isFinite,
                number <= Double(Int.max),
                number >= Double(Int.min)
            {
                return Int(number)
            }
            return nil
        default:
            return nil
        }
    }

    private func nestedIntValue(in json: [String: Any], path: [String]) -> Int? {
        var current: Any = json
        for key in path {
            guard let dict = current as? [String: Any], let next = dict[key] else {
                return nil
            }
            current = next
        }
        return intValue(from: current)
    }

    private func loadJSON(at url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }
        return json
    }

    private func validateGemma3nMultimodalWeights(model: AIModel, modelDir: URL) throws {
        guard model.name.localizedCaseInsensitiveContains("Gemma-3n") else { return }

        let indexURL = modelDir.appendingPathComponent("model.safetensors.index.json")
        guard
            let json = loadJSON(at: indexURL),
            let weightMap = json["weight_map"] as? [String: Any]
        else {
            return
        }

        let keys = weightMap.keys
        let hasAudioTower = keys.contains { $0.hasPrefix("model.audio_tower.") }
        let hasVisionTower = keys.contains { $0.hasPrefix("model.vision_tower.") || $0.hasPrefix("model.embed_vision.") }

        if model.supportsAudio && !hasAudioTower {
            throw NSError(
                domain: "LLMBackend",
                code: 422,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Gemma-3n files are not the full multimodal weights (audio tower missing). Delete this model and re-download from the current model card."
                ]
            )
        }

        if model.supportsVision && !hasVisionTower {
            throw NSError(
                domain: "LLMBackend",
                code: 422,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Gemma-3n files are not the full multimodal weights (vision tower missing). Delete this model and re-download from the current model card."
                ]
            )
        }
    }

    private func resolveContextWindowSize(modelDir: URL) -> Int {
        let candidateFiles = [
            "config.json",
            "tokenizer_config.json",
            "generation_config.json"
        ]

        let directKeys = [
            "max_position_embeddings",
            "n_positions",
            "max_seq_len",
            "max_sequence_length",
            "seq_len",
            "seq_length",
            "context_length",
            "model_max_length"
        ]

        let nestedPaths: [[String]] = [
            ["text_config", "max_position_embeddings"],
            ["text_config", "n_positions"],
            ["text_config", "model_max_length"],
            ["llm_config", "max_position_embeddings"],
            ["llm_config", "context_length"],
            ["rope_scaling", "original_max_position_embeddings"],
            ["rope_scaling", "max_position_embeddings"]
        ]

        var discovered: [Int] = []

        for fileName in candidateFiles {
            let url = modelDir.appendingPathComponent(fileName)
            guard let json = loadJSON(at: url) else { continue }

            for key in directKeys {
                if let value = intValue(from: json[key]), value > 0 {
                    discovered.append(value)
                }
            }

            for path in nestedPaths {
                if let value = nestedIntValue(in: json, path: path), value > 0 {
                    discovered.append(value)
                }
            }
        }

        // Remove obvious placeholder/outlier values and choose the largest plausible context.
        let plausible = discovered.filter { $0 >= 128 && $0 <= 1_048_576 }
        if let resolved = plausible.max() {
            return resolved
        }

        return 2048
    }

    private func shouldStopForDegenerateLoop(_ text: String) -> Bool {
        guard text.count >= 180 else { return false }

        // Detect repeated trailing pattern (common runaway loop symptom).
        let tail = String(text.suffix(180))
        let slice = 60
        let chunks = stride(from: 0, to: tail.count, by: slice).compactMap { idx -> String? in
            let start = tail.index(tail.startIndex, offsetBy: idx)
            let end = tail.index(start, offsetBy: min(slice, tail.count - idx), limitedBy: tail.endIndex) ?? tail.endIndex
            let part = String(tail[start..<end])
            return part.count == slice ? part : nil
        }

        if chunks.count >= 3 {
            let c1 = chunks[chunks.count - 1]
            let c2 = chunks[chunks.count - 2]
            let c3 = chunks[chunks.count - 3]
            if c1 == c2 && c2 == c3 {
                return true
            }
        }

        return false
    }

    private func sanitizeChunk(_ chunk: String) -> (text: String, shouldStop: Bool) {
        var text = chunk
        var shouldStop = false

        for marker in stopMarkers {
            if let range = text.range(of: marker) {
                text = String(text[..<range.lowerBound])
                shouldStop = true
            }
        }

        return (text, shouldStop)
    }
    
    func loadModel(_ model: AIModel) async throws {
        isBackendLoading = true
        defer { isBackendLoading = false }

        print("[LLMBackend] loadModel name=\(model.name) visionEnabled=\(enableVision) audioEnabled=\(enableAudio)")
        
        let fileManager = FileManager.default
        let documentsDir = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        let modelsDir = documentsDir.appendingPathComponent("models")
        let modelDir = modelsDir.appendingPathComponent(model.id)
        
        // 1. Verify files exist
        for file in model.files {
            let path = modelDir.appendingPathComponent(file).path
            if !fileManager.fileExists(atPath: path) {
                throw NSError(domain: "LLMBackend", code: 404, userInfo: [NSLocalizedDescriptionKey: "Missing model file: \(file)"])
            }
        }

        try validateGemma3nMultimodalWeights(model: model, modelDir: modelDir)
        
        // 2. Load with MLX LLM
        var modelConfiguration = ModelConfiguration(
            directory: modelDir,
            extraEOSTokens: Set(stopMarkers)
        )
        modelConfiguration.extraEOSTokens.formUnion(stopMarkers)
        let hub = HubApi()
        let factory = LLMModelFactory.shared
        do {
            self.modelContainer = try await factory.loadContainer(hub: hub, configuration: modelConfiguration)
        } catch {
            let description = String(describing: error)
            print("[LLMBackend] loadModel rawError=\(description)")
            if description.contains("altup_unembed_projections") || description.contains("keyNotFound(path") {
                throw NSError(
                    domain: "LLMBackend",
                    code: 422,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "Model files are incomplete/corrupted or incompatible with current MLX runtime. Delete this model and download again. (\(description))"
                    ]
                )
            }
            throw error
        }
        self.currentContextWindowSize = max(512, resolveContextWindowSize(modelDir: modelDir))
        
        isLoaded = true
        currentlyLoadedModel = model.name
    }
    
    func unloadModel() {
        modelContainer = nil
        isLoaded = false
        currentlyLoadedModel = nil
    }
    
    func generate(
        prompt: String,
        imageURL: URL? = nil,
        audioURL: URL? = nil,
        onUpdate: @escaping (String, Int, Double) -> Void
    ) async throws {
        guard let container = self.modelContainer else { 
            throw NSError(domain: "LLMBackend", code: 403, userInfo: [NSLocalizedDescriptionKey: "Model not loaded"]) 
        }

        var images: [UserInput.Image] = []
        var videos: [UserInput.Video] = []

        if enableVision, let imageURL {
            images.append(.url(imageURL))
        }
        if enableAudio, let audioURL {
            // MLX UserInput exposes media as videos for AVAsset-backed processing.
            videos.append(.url(audioURL))
        }

        print(
            "[LLMBackend] generate visionEnabled=\(enableVision) audioEnabled=\(enableAudio) images=\(images.count) videos=\(videos.count)"
        )

        let input = try await container.prepare(
            input: UserInput(prompt: prompt, images: images, videos: videos)
        )
        // Respect the user setting, with model max context as the only ceiling.
        let modelContextCeiling = max(32, self.currentContextWindowSize)
        let effectiveMaxTokens = max(32, min(self.maxTokens, modelContextCeiling))

        let params = GenerateParameters(
            maxTokens: effectiveMaxTokens,
            temperature: self.temperature,
            topP: self.topP,
            repetitionPenalty: 1.10,
            repetitionContextSize: 128
        )
        
        let startTime = Date()
        var currentOutput = ""
        var tokens = 0
        
        let stream = try await container.generate(input: input, parameters: params)
        for await item in stream {
            if Task.isCancelled { break }
            switch item {
            case .chunk(let text):
                let sanitized = sanitizeChunk(text)
                if !sanitized.text.isEmpty {
                    currentOutput += sanitized.text
                    tokens += 1
                    let elapsed = Date().timeIntervalSince(startTime)
                    let tps = elapsed > 0 ? Double(tokens) / elapsed : 0
                    onUpdate(currentOutput, tokens, tps)
                }
                if sanitized.shouldStop {
                    break
                }
                if shouldStopForDegenerateLoop(currentOutput) {
                    break
                }
            case .info:
                return
            case .toolCall:
                break
            }
        }
    }
}
