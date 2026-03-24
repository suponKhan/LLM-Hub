import SwiftUI
import RunAnywhere

struct ModelFamilyGroup: Identifiable {
    let title: String
    let models: [AIModel]
    var id: String { title }
}

private extension URLError.Code {
    var isTransientDownloadFailure: Bool {
        switch self {
        case .networkConnectionLost,
            .notConnectedToInternet,
            .timedOut,
            .cannotConnectToHost,
            .cannotFindHost,
            .dnsLookupFailed,
            .resourceUnavailable,
            .internationalRoamingOff,
            .callIsActive,
            .dataNotAllowed:
            return true
        default:
            return false
        }
    }
}

// MARK: - Download ViewModel
@MainActor
class ModelDownloadViewModel: ObservableObject {
    @Published var models: [AIModel] = ModelData.models
    @Published var selectedCategory: ModelCategory = .multimodal
    @Published var searchText: String = ""
    @Published var downloadStates: [String: DownloadState] = [:]
    @Published var expandedModelId: String? = nil
    @Published var expandedFamilyTitle: String? = nil
    private let completionThresholdRatio: Double = 0.98
    private let pendingDownloadsKey = "ios_pending_model_download_ids"

    private func legacyModelDirectory(for model: AIModel) -> URL? {
        guard let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return nil }
        return documentsDir.appendingPathComponent("models").appendingPathComponent(model.id)
    }

    private func destinationDirectory(for model: AIModel) throws -> URL {
        try SimplifiedFileManager.shared.getModelFolderURL(modelId: model.id, framework: model.inferenceFramework)
    }

    private func requiredFilesExist(in directory: URL, for model: AIModel) -> (allExist: Bool, totalBytes: Int64) {
        var allExist = true
        var totalLocalBytes: Int64 = 0

        if model.requiredFileNames.isEmpty {
            return (false, 0)
        }

        for fileName in model.requiredFileNames {
            let filePath = directory.appendingPathComponent(fileName)
            if !FileManager.default.fileExists(atPath: filePath.path) {
                allExist = false
                break
            }
            if let attrs = try? FileManager.default.attributesOfItem(atPath: filePath.path),
               let fileSize = attrs[.size] as? Int64 {
                totalLocalBytes += fileSize
            }
        }

        return (allExist, totalLocalBytes)
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

        // Initialize with default states
        for model in ModelData.models {
            downloadStates[model.id] = .notDownloaded
        }
        refreshStatuses()
        resumePendingDownloads()
    }

    private func loadPendingDownloadIDs() -> Set<String> {
        let ids = UserDefaults.standard.stringArray(forKey: pendingDownloadsKey) ?? []
        return Set(ids)
    }

    private func savePendingDownloadIDs(_ ids: Set<String>) {
        UserDefaults.standard.set(Array(ids), forKey: pendingDownloadsKey)
    }

    private func markPending(_ id: String) {
        var ids = loadPendingDownloadIDs()
        ids.insert(id)
        savePendingDownloadIDs(ids)
    }

    private func clearPending(_ id: String) {
        var ids = loadPendingDownloadIDs()
        ids.remove(id)
        savePendingDownloadIDs(ids)
    }

    func refreshStatuses() {
        for model in models {
            var allExist = false
            var totalLocalBytes: Int64 = 0

            if let runAnywhereDir = try? destinationDirectory(for: model),
               FileManager.default.fileExists(atPath: runAnywhereDir.path) {
                let status = requiredFilesExist(in: runAnywhereDir, for: model)
                allExist = status.allExist
                totalLocalBytes = status.totalBytes
            }

            if !allExist, let legacyDir = legacyModelDirectory(for: model), FileManager.default.fileExists(atPath: legacyDir.path) {
                let status = requiredFilesExist(in: legacyDir, for: model)
                allExist = status.allExist
                totalLocalBytes = max(totalLocalBytes, status.totalBytes)
            }
            
            let minimumExpectedBytes = Int64(Double(model.sizeBytes) * completionThresholdRatio)
            if allExist && totalLocalBytes >= minimumExpectedBytes {
                self.downloadStates[model.id] = .downloaded
            } else {
                // If it's currently downloading in this session, don't overwrite
                if case .downloading = downloadStates[model.id] {
                    continue
                }
                self.downloadStates[model.id] = totalLocalBytes > 0 ? .paused : .notDownloaded
            }
        }
    }

    var filteredModels: [AIModel] {
        let categoryFiltered = models.filter { $0.category == selectedCategory }
        if searchText.isEmpty { return categoryFiltered }
        return categoryFiltered.filter { $0.name.localizedCaseInsensitiveContains(searchText) || $0.description.localizedCaseInsensitiveContains(searchText) }
    }

    var groupedFilteredModels: [ModelFamilyGroup] {
        let grouped = Dictionary(grouping: filteredModels, by: familyName(for:))
        return grouped
            .map { key, value in
                let sortedModels = value.sorted { lhs, rhs in
                    quantizationLabel(for: lhs).localizedCaseInsensitiveCompare(quantizationLabel(for: rhs)) == .orderedAscending
                }
                return ModelFamilyGroup(title: key, models: sortedModels)
            }
            .sorted { $0.title.localizedCaseInsensitiveCompare($1.title) == .orderedAscending }
    }

    func toggleFamily(_ title: String) {
        if expandedFamilyTitle == title {
            expandedFamilyTitle = nil
        } else {
            expandedFamilyTitle = title
        }
    }

    func quantizationLabel(for model: AIModel) -> String {
        guard let openIndex = model.name.lastIndex(of: "("),
              let closeIndex = model.name.lastIndex(of: ")"),
              openIndex < closeIndex else {
            return model.name
        }
        return String(model.name[model.name.index(after: openIndex)..<closeIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func familyName(for model: AIModel) -> String {
        guard let openIndex = model.name.lastIndex(of: "("),
              let closeIndex = model.name.lastIndex(of: ")"),
              openIndex < closeIndex else {
            return model.name
        }

        let suffix = model.name[model.name.index(after: openIndex)..<closeIndex]
        let suffixUpper = suffix.uppercased().replacingOccurrences(of: "-", with: "_")
        let likelyQuant = suffixUpper.contains("Q") || suffixUpper.contains("IQ") || suffixUpper.contains("F16") || suffixUpper.contains("BF16")
        if likelyQuant {
            return model.name[..<openIndex].trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return model.name
    }

    func toggleExpand(_ id: String) {
        if expandedModelId == id {
            expandedModelId = nil
        } else {
            expandedModelId = id
        }
    }

    private var downloadTasks: [String: Task<Void, Never>] = [:]
    private var autoResumableDownloads: Set<String> = []

    func startDownload(_ model: AIModel) {
        // Cancel existing task if any
        downloadTasks[model.id]?.cancel()
        markPending(model.id)
        
        let task = Task {
            do {
                try RunAnywhere.initialize(environment: .development)
            } catch {
                // Initialization may already be in progress/complete in other flows.
            }

            let destinationDir: URL
            do {
                destinationDir = try destinationDirectory(for: model)
            } catch {
                await MainActor.run {
                    self.downloadStates[model.id] = .error(message: error.localizedDescription)
                    self.clearPending(model.id)
                }
                return
            }
            
            await MainActor.run {
                downloadStates[model.id] = .downloading(progress: 0, downloaded: "0 MB", speed: "0 KB/s")
            }
            
            do {
                try await ModelDownloader.shared.downloadModel(
                    model,
                    hfToken: nil,
                    destinationDir: destinationDir,
                    onProgress: { update in
                        Task { @MainActor in
                            let downloadedLabel = ByteCountFormatter.string(fromByteCount: update.bytesDownloaded, countStyle: .file)
                            let speedLabel = ByteCountFormatter.string(fromByteCount: Int64(update.speedBytesPerSecond), countStyle: .file) + "/s"
                            let progress = Double(update.bytesDownloaded) / Double(max(1, update.totalBytes))
                            self.downloadStates[model.id] = .downloading(progress: progress, downloaded: downloadedLabel, speed: speedLabel)
                        }
                    }
                )
                
                await MainActor.run {
                    self.downloadStates[model.id] = .downloaded
                    self.downloadTasks.removeValue(forKey: model.id)
                    self.clearPending(model.id)
                    self.refreshStatuses()
                }

                _ = await RunAnywhere.discoverDownloadedModels()
            } catch is CancellationError {
                await MainActor.run {
                    self.downloadStates[model.id] = .paused
                }
            } catch let error as URLError where error.code == .cancelled {
                await MainActor.run {
                    self.downloadStates[model.id] = .paused
                }
            } catch let error as URLError where error.code.isTransientDownloadFailure {
                await MainActor.run {
                    // Treat temporary connection drops as recoverable; avoid surfacing noisy errors.
                    self.downloadStates[model.id] = .paused
                    self.autoResumableDownloads.insert(model.id)
                }
            } catch let error as NSError where error.domain == "ModelDownloader" && error.code == -2 {
                await MainActor.run {
                    // Incomplete file set after interruption is recoverable by resuming.
                    self.downloadStates[model.id] = .paused
                    self.autoResumableDownloads.insert(model.id)
                }
            } catch {
                await MainActor.run {
                    self.downloadStates[model.id] = .error(message: error.localizedDescription)
                    self.clearPending(model.id)
                }
            }
        }
        downloadTasks[model.id] = task
    }

    func pauseDownload(_ id: String) {
        downloadTasks[id]?.cancel()
        downloadTasks.removeValue(forKey: id)
        downloadStates[id] = .paused
        clearPending(id)
    }

    func resumeDownload(_ id: String) {
        if let model = models.first(where: { $0.id == id }) {
            autoResumableDownloads.remove(id)
            startDownload(model)
        }
    }

    func resumeAutoResumableDownloads() {
        let ids = autoResumableDownloads
        autoResumableDownloads.removeAll()
        for id in ids {
            if case .paused = downloadStates[id], let model = models.first(where: { $0.id == id }) {
                startDownload(model)
            }
        }
    }

    func resumePendingDownloads() {
        let ids = loadPendingDownloadIDs()
        for id in ids {
            guard downloadTasks[id] == nil,
                  let model = models.first(where: { $0.id == id }) else {
                continue
            }

            switch downloadStates[id] {
            case .downloaded:
                clearPending(id)
            case .downloading:
                continue
            case .paused, .notDownloaded, .error, .none:
                startDownload(model)
            }
        }
    }

    func deleteModel(_ id: String) {
        downloadTasks[id]?.cancel()
        downloadTasks.removeValue(forKey: id)
        clearPending(id)
        
        let model = models.first(where: { $0.id == id })
        if let model = model {
            if let destinationDir = try? destinationDirectory(for: model) {
                try? FileManager.default.removeItem(at: destinationDir)
            }
            if let legacyDir = legacyModelDirectory(for: model) {
                try? FileManager.default.removeItem(at: legacyDir)
            }
            downloadStates[id] = .notDownloaded
        }
    }
}

// MARK: - Model Row View
struct ModelRowView: View {
    @EnvironmentObject var settings: AppSettings
    let model: AIModel
    let state: DownloadState
    let isExpanded: Bool
    let onDownload: () -> Void
    let onPause: () -> Void
    let onResume: () -> Void
    let onDelete: () -> Void
    let onExpand: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            Button(action: onExpand) {
                HStack(spacing: 12) {
                    // Model icon
                    ZStack {
                        RoundedRectangle(cornerRadius: 10)
                            .fill(categoryGradient)
                            .frame(width: 44, height: 44)
                        Image(systemName: model.category.icon)
                            .font(.system(size: 20))
                            .foregroundColor(.white)
                    }

                    VStack(alignment: .leading, spacing: 3) {
                        Text(model.name)
                            .font(.subheadline.bold())
                            .foregroundColor(.primary)
                            .lineLimit(2)

                        HStack(spacing: 6) {
                            StatusBadge(state: state)
                            Text("•")
                                .foregroundColor(.secondary)
                            Text(model.sizeLabel)
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("•")
                                .foregroundColor(.secondary)
                            Text(String(format: settings.localized("ram_requirement_format"), Int(model.requirements.minRamGB)))
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }

                        // Capability badges
                        HStack(spacing: 4) {
                            if model.supportsVision {
                                capabilityBadge(settings.localized("vision"), color: .purple)
                            }
                            if model.supportsAudio {
                                capabilityBadge(settings.localized("audio"), color: .orange)
                            }
                            if !model.supportsVision && !model.supportsAudio {
                                capabilityBadge(settings.localized("text_only"), color: .indigo)
                            }
                        }
                    }

                    Spacer()

                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.vertical, 12)
                .padding(.horizontal, 16)
                .contentShape(Rectangle()) // Makes the whole area clickable
            }
            .buttonStyle(.plain)

            // Download progress
            if case .downloading(let progress, let downloaded, let speed) = state {
                VStack(spacing: 4) {
                    ProgressView(value: progress)
                        .tint(.indigo)
                        .padding(.horizontal, 16)
                    HStack {
                        Text("\(settings.localized("downloading")) \(downloaded) / \(model.sizeLabel) (\(speed))")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Spacer()
                        Text("\(Int(progress * 100))%")
                            .font(.caption.bold())
                            .foregroundColor(.indigo)
                    }
                    .padding(.horizontal, 16)
                }
                .padding(.bottom, 8)
            }

            // Expanded details
            if isExpanded {
                VStack(alignment: .leading, spacing: 12) {
                    Divider()
                        .padding(.horizontal, 16)


                    // Model description removed per user request


                    HStack(spacing: 6) {
                        Image(systemName: "link")
                            .font(.caption)
                            .foregroundColor(.indigo)
                        Text(model.url)
                            .font(.caption)
                            .foregroundColor(.indigo)
                            .lineLimit(1)
                    }
                    .padding(.horizontal, 16)

                    // Action buttons
                    HStack(spacing: 10) {
                        switch state {
                        case .notDownloaded:
                            Button(action: onDownload) {
                                Label(settings.localized("download"), systemImage: "arrow.down.circle.fill")
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 10)
                                    .background(.indigo.gradient)
                                    .foregroundColor(.white)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                            }
                        case .error:
                            Button(action: onDownload) {
                                Label(settings.localized("retry"), systemImage: "arrow.clockwise")
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 10)
                                    .background(.red.gradient)
                                    .foregroundColor(.white)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                            }
                        case .downloading:
                            Button(action: onPause) {
                                Label(settings.localized("pause_download"), systemImage: "pause.circle.fill")
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 10)
                                    .background(.orange.gradient)
                                    .foregroundColor(.white)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                            }
                            Button(action: onDelete) {
                                Image(systemName: "xmark")
                                    .padding(.vertical, 10)
                                    .padding(.horizontal, 14)
                                    .background(Color.red.opacity(0.1))
                                    .foregroundColor(.red)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                            }
                        case .paused:
                            Button(action: onResume) {
                                Label(settings.localized("resume_download"), systemImage: "play.circle.fill")
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 10)
                                    .background(.green.gradient)
                                    .foregroundColor(.white)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                            }
                            Button(action: onDelete) {
                                Image(systemName: "trash")
                                    .padding(.vertical, 10)
                                    .padding(.horizontal, 14)
                                    .background(Color.red.opacity(0.1))
                                    .foregroundColor(.red)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                            }
                        case .downloaded:
                            Button(action: onDelete) {
                                Label(settings.localized("action_delete"), systemImage: "trash")
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 10)
                                    .background(.red.gradient)
                                    .foregroundColor(.white)
                                    .clipShape(RoundedRectangle(cornerRadius: 10))
                            }
                        }
                    }
                    .font(.subheadline.bold())
                    .padding(.horizontal, 16)
                }
                .padding(.bottom, 12)
            }
        }
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    private var categoryGradient: LinearGradient {
        switch model.category {
        case .text:       return LinearGradient(colors: [.indigo, .blue], startPoint: .topLeading, endPoint: .bottomTrailing)
        case .multimodal: return LinearGradient(colors: [.purple, .pink], startPoint: .topLeading, endPoint: .bottomTrailing)
        case .embedding:  return LinearGradient(colors: [.teal, .cyan], startPoint: .topLeading, endPoint: .bottomTrailing)
        case .imageGeneration: return LinearGradient(colors: [.orange, .red], startPoint: .topLeading, endPoint: .bottomTrailing)
        }
    }

    private func capabilityBadge(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.caption2)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.15))
            .foregroundColor(color)
            .clipShape(Capsule())
    }
}

// MARK: - Status Badge
struct StatusBadge: View {
    @EnvironmentObject var settings: AppSettings
    let state: DownloadState

    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(dotColor)
                .frame(width: 6, height: 6)
            Text(label)
                .font(.caption)
                .foregroundColor(dotColor)
        }
    }

    private var label: String {
        switch state {
        case .notDownloaded: return settings.localized("not_downloaded")
        case .downloading: return settings.localized("downloading")
        case .paused: return settings.localized("paused")
        case .downloaded: return settings.localized("downloaded")
        case .error: return settings.localized("error")
        }
    }

    private var dotColor: Color {
        switch state {
        case .notDownloaded:  return .secondary
        case .downloading:    return .indigo
        case .paused:         return .orange
        case .downloaded:     return .green
        case .error:          return .red
        }
    }
}

// MARK: - ModelDownloadScreen
struct ModelDownloadScreen: View {
    @EnvironmentObject var settings: AppSettings
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var vm = ModelDownloadViewModel()
    var onNavigateBack: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            // Category Tabs
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(ModelCategory.allCases, id: \.self) { cat in
                        CategoryTab(
                            category: cat,
                            isSelected: vm.selectedCategory == cat,
                            count: vm.models.filter { $0.category == cat }.count
                        ) {
                            withAnimation(.spring(response: 0.3)) {
                                vm.selectedCategory = cat
                            }
                        }
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
            }
            .background(.thinMaterial)



            // Model list
            ScrollView {
                LazyVStack(spacing: 12) {
                    if vm.filteredModels.isEmpty {
                        VStack(spacing: 16) {
                            Image(systemName: "magnifyingglass")
                                .font(.system(size: 48))
                                .foregroundColor(.secondary.opacity(0.5))
                            Text(settings.localized("no_models_available"))
                                .foregroundColor(.secondary)
                        }
                        .padding(.top, 60)
                    } else {
                        ForEach(vm.groupedFilteredModels) { family in
                            VStack(alignment: .leading, spacing: 10) {
                                Button {
                                    withAnimation(.spring(response: 0.28, dampingFraction: 0.85)) {
                                        vm.toggleFamily(family.title)
                                    }
                                } label: {
                                    HStack(spacing: 10) {
                                        Image(systemName: "folder.fill")
                                            .foregroundColor(.indigo)
                                        Text(family.title)
                                            .font(.subheadline.bold())
                                            .foregroundColor(.primary)
                                        Text("(\(family.models.count))")
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                        Spacer()
                                        Image(systemName: vm.expandedFamilyTitle == family.title ? "chevron.up" : "chevron.down")
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    }
                                    .padding(.horizontal, 14)
                                    .padding(.vertical, 12)
                                    .background(Color(.secondarySystemBackground))
                                    .clipShape(RoundedRectangle(cornerRadius: 12))
                                }
                                .buttonStyle(.plain)

                                if vm.expandedFamilyTitle == family.title {
                                    ForEach(family.models) { model in
                                        VStack(alignment: .leading, spacing: 6) {
                                            Text(vm.quantizationLabel(for: model))
                                                .font(.caption.bold())
                                                .foregroundColor(.indigo)
                                                .padding(.horizontal, 4)

                                            ModelRowView(
                                                model: model,
                                                state: vm.downloadStates[model.id] ?? .notDownloaded,
                                                isExpanded: vm.expandedModelId == model.id,
                                                onDownload: { vm.startDownload(model) },
                                                onPause:    { vm.pauseDownload(model.id) },
                                                onResume:   { vm.resumeDownload(model.id) },
                                                onDelete:   { vm.deleteModel(model.id) },
                                                onExpand:   { vm.toggleExpand(model.id) }
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                .padding(.horizontal, 16)
                .padding(.top, 24)
                .padding(.bottom, 24)
            }
        }
        .background(Color(.systemGroupedBackground))
        .navigationTitle(settings.localized("ai_models"))
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button {
                    onNavigateBack()
                } label: {
                    Image(systemName: "arrow.left")
                        .font(.headline)
                }
            }
        }
        .onAppear {
            vm.refreshStatuses()
            vm.resumePendingDownloads()
        }
        .onChange(of: scenePhase) { _, newPhase in
            if newPhase == .active {
                vm.resumeAutoResumableDownloads()
                vm.resumePendingDownloads()
            }
        }
    }
}

// MARK: - Category Tab
struct CategoryTab: View {
    @EnvironmentObject var settings: AppSettings
    let category: ModelCategory
    let isSelected: Bool
    let count: Int
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 6) {
                Image(systemName: category.icon)
                    .font(.subheadline)
                Text(settings.localized(category.titleKey))
                    .font(.subheadline.bold())
                Text("\(count)")
                    .font(.caption)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(isSelected ? Color.white.opacity(0.25) : Color.secondary.opacity(0.15))
                    .foregroundColor(isSelected ? .white : .secondary)
                    .clipShape(Capsule())
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(isSelected ? AnyShapeStyle(.indigo.gradient) : AnyShapeStyle(Color(.secondarySystemBackground)))
            .foregroundColor(isSelected ? .white : .primary)
            .clipShape(RoundedRectangle(cornerRadius: 10))
        }
        .buttonStyle(.plain)
    }
}

