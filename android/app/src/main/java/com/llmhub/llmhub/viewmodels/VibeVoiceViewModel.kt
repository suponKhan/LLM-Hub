package com.llmhub.llmhub.viewmodels

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelAvailabilityProvider
import com.llmhub.llmhub.inference.MediaPipeInferenceService
import com.llmhub.llmhub.utils.AudioConversionUtils
import java.util.UUID
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch

class VibeVoiceViewModel(application: Application) : AndroidViewModel(application) {
    companion object {
        private const val VIBEVOICE_SYSTEM_PROMPT = """
            You are VibeVoice, a natural real-time conversation assistant.
            Keep responses conversational and useful.
            Match response length to the user's request.
            For simple requests, keep it brief.
            For "why/how/explain/compare/steps" requests, give a fuller multi-sentence answer.
            Respond to the meaning of what the user said in audio.
            Do not transcribe or repeat the user's words verbatim.
            Do not output role labels like 'assistant:' or 'user:'.
            If audio is unclear, ask one brief clarification question.
        """

        private const val VOICE_TURN_PROMPT = """
            The user just spoke in audio.
            Reply naturally and match the detail level the user asked for.
            Do not transcribe or quote the user's exact words.
        """

        private const val VOICE_FALLBACK_REPLY = "I heard you, but I missed part of that. Could you repeat it briefly?"
    }

    private val inferenceService = MediaPipeInferenceService(application)
    private val prefs = application.getSharedPreferences("vibevoice_prefs", android.content.Context.MODE_PRIVATE)

    data class VoiceTurn(
        val id: String,
        val assistantText: String
    )

    private val _availableModels = MutableStateFlow<List<LLMModel>>(emptyList())
    val availableModels: StateFlow<List<LLMModel>> = _availableModels.asStateFlow()

    private val _selectedModel = MutableStateFlow<LLMModel?>(null)
    val selectedModel: StateFlow<LLMModel?> = _selectedModel.asStateFlow()

    private val _selectedBackend = MutableStateFlow(LlmInference.Backend.GPU)
    val selectedBackend: StateFlow<LlmInference.Backend> = _selectedBackend.asStateFlow()

    private val _selectedNpuDeviceId = MutableStateFlow<String?>(null)
    val selectedNpuDeviceId: StateFlow<String?> = _selectedNpuDeviceId.asStateFlow()

    private val _isLoadingModel = MutableStateFlow(false)
    val isLoadingModel: StateFlow<Boolean> = _isLoadingModel.asStateFlow()

    private val _isModelLoaded = MutableStateFlow(false)
    val isModelLoaded: StateFlow<Boolean> = _isModelLoaded.asStateFlow()

    private val _isRecording = MutableStateFlow(false)
    val isRecording: StateFlow<Boolean> = _isRecording.asStateFlow()

    private val _isResponding = MutableStateFlow(false)
    val isResponding: StateFlow<Boolean> = _isResponding.asStateFlow()

    private val _loadError = MutableStateFlow<String?>(null)
    val loadError: StateFlow<String?> = _loadError.asStateFlow()

    private val _audioUri = MutableStateFlow<Uri?>(null)
    val audioUri: StateFlow<Uri?> = _audioUri.asStateFlow()

    private val _audioData = MutableStateFlow<ByteArray?>(null)
    val audioData: StateFlow<ByteArray?> = _audioData.asStateFlow()

    private val _liveResponseText = MutableStateFlow("")
    val liveResponseText: StateFlow<String> = _liveResponseText.asStateFlow()

    private val _conversationTurns = MutableStateFlow<List<VoiceTurn>>(emptyList())
    val conversationTurns: StateFlow<List<VoiceTurn>> = _conversationTurns.asStateFlow()

    private var respondJob: Job? = null
    private var sessionChatId: String = "vibevoice-${UUID.randomUUID()}"
    private var isSessionPrimed: Boolean = false

    init {
        loadAvailableModels()
        loadSavedSettings()
    }

    private fun loadAvailableModels() {
        viewModelScope.launch {
            val context = getApplication<Application>()
            val allModels = ModelAvailabilityProvider.loadAvailableModels(context)
            val audioModels = allModels.filter { it.supportsAudio }
            _availableModels.value = audioModels

            val savedModelName = prefs.getString("selected_model_name", null)
            if (savedModelName != null) {
                val saved = audioModels.find { it.name == savedModelName }
                if (saved != null) _selectedModel.value = saved
            }
            if (audioModels.isNotEmpty() && _selectedModel.value == null) {
                _selectedModel.value = audioModels.first()
            }
        }
    }

    private fun loadSavedSettings() {
        val savedBackendName = prefs.getString("selected_backend", LlmInference.Backend.GPU.name)
        _selectedBackend.value = try {
            LlmInference.Backend.valueOf(savedBackendName ?: LlmInference.Backend.GPU.name)
        } catch (_: IllegalArgumentException) {
            LlmInference.Backend.GPU
        }
    }

    private fun saveSettings() {
        prefs.edit().apply {
            putString("selected_model_name", _selectedModel.value?.name)
            putString("selected_backend", _selectedBackend.value.name)
            apply()
        }
    }

    fun selectModel(model: LLMModel) {
        if (_isModelLoaded.value) unloadModel()
        _selectedModel.value = model
        saveSettings()
    }

    fun selectBackend(backend: LlmInference.Backend, deviceId: String? = null) {
        if (_isModelLoaded.value) unloadModel()
        _selectedBackend.value = backend
        _selectedNpuDeviceId.value = deviceId
        saveSettings()
    }

    fun setAudioUri(uri: Uri?) {
        _audioUri.value = uri
        _audioData.value = null
    }

    fun setAudioData(data: ByteArray?) {
        _audioData.value = data
        if (data != null) _audioUri.value = null
    }

    fun setRecording(recording: Boolean) {
        _isRecording.value = recording
    }

    fun clearError() {
        _loadError.value = null
    }

    fun loadModel() {
        val model = _selectedModel.value ?: return
        if (_isLoadingModel.value || _isModelLoaded.value) return

        viewModelScope.launch {
            _isLoadingModel.value = true
            _loadError.value = null
            try {
                inferenceService.loadModel(
                    model = model,
                    preferredBackend = _selectedBackend.value,
                    disableVision = true,
                    disableAudio = false,
                    deviceId = _selectedNpuDeviceId.value
                )
                _isModelLoaded.value = true
            } catch (e: Exception) {
                _loadError.value = e.message ?: "Failed to load model"
                _isModelLoaded.value = false
            } finally {
                _isLoadingModel.value = false
            }
        }
    }

    fun unloadModel() {
        viewModelScope.launch {
            try {
                respondJob?.cancel()
                inferenceService.unloadModel()
                _isModelLoaded.value = false
            } catch (e: Exception) {
                _loadError.value = e.message ?: "Failed to unload model"
            }
        }
    }

    fun cancelResponse() {
        respondJob?.cancel()
        respondJob = null
        _isResponding.value = false
        _liveResponseText.value = ""

        // Ensure the underlying MediaPipe generation is also cancelled so it cannot resume later.
        viewModelScope.launch {
            try {
                inferenceService.resetChatSession(sessionChatId)
                isSessionPrimed = false
            } catch (_: Exception) {
                // Best-effort cancel.
            }
        }
    }

    fun sendVoiceTurn(audioUri: Uri? = null) {
        val model = _selectedModel.value ?: return
        respondJob?.cancel()

        respondJob = viewModelScope.launch {
            _isResponding.value = true
            _liveResponseText.value = ""

            try {
                val uriToUse = audioUri ?: _audioUri.value
                val audioBytes = _audioData.value ?: (uriToUse?.let { readAudioBytes(it) })
                    ?: throw IllegalStateException("Unable to read audio input")

                val turnId = UUID.randomUUID().toString()
                _conversationTurns.value = _conversationTurns.value + VoiceTurn(
                    id = turnId,
                    assistantText = ""
                )

                val turnPrompt = if (!isSessionPrimed) {
                    isSessionPrimed = true
                    "system: $VIBEVOICE_SYSTEM_PROMPT\n\nuser: $VOICE_TURN_PROMPT"
                } else {
                    "user: $VOICE_TURN_PROMPT"
                }

                val responseFlow = inferenceService.generateResponseStreamWithSession(
                    prompt = turnPrompt,
                    model = model,
                    chatId = sessionChatId,
                    images = emptyList(),
                    audioData = audioBytes,
                    webSearchEnabled = false
                )

                responseFlow.collect { token ->
                    if (token.isNotEmpty()) {
                        _liveResponseText.value += token
                        _conversationTurns.value = _conversationTurns.value.map {
                            if (it.id == turnId) it.copy(assistantText = _liveResponseText.value) else it
                        }
                    }
                }

                val cleaned = sanitizeVoiceResponse(_liveResponseText.value)
                val finalReply = if (cleaned.isBlank() || isPromptEcho(cleaned)) {
                    VOICE_FALLBACK_REPLY
                } else {
                    cleaned
                }

                _liveResponseText.value = finalReply
                _conversationTurns.value = _conversationTurns.value.map {
                    if (it.id == turnId) it.copy(assistantText = finalReply) else it
                }
            } catch (_: CancellationException) {
                // Ignore cancellations.
            } catch (e: Exception) {
                // Keep one ongoing session for normal turns, but recover automatically if context is full.
                if (isContextLimitError(e)) {
                    try {
                        inferenceService.resetChatSession(sessionChatId)
                    } catch (_: Exception) {
                        // Best-effort reset.
                    }
                    sessionChatId = "vibevoice-${UUID.randomUUID()}"
                    isSessionPrimed = false
                    _loadError.value = "Session context reached limit. Started a fresh voice session."
                } else {
                    _loadError.value = e.message ?: "Voice chat failed"
                }
            } finally {
                _isResponding.value = false
            }
        }
    }

    fun clearCurrentAudioInput() {
        _audioData.value = null
        _audioUri.value = null
    }

    fun clearConversation() {
        _conversationTurns.value = emptyList()
        _liveResponseText.value = ""
        sessionChatId = "vibevoice-${UUID.randomUUID()}"
        isSessionPrimed = false
    }

    private fun isContextLimitError(error: Throwable): Boolean {
        val msg = (error.message ?: "").lowercase()
        return msg.contains("out_of_range") ||
            msg.contains("context") ||
            msg.contains("token") ||
            msg.contains("exceed") ||
            msg.contains("capacity") ||
            msg.contains("limit")
    }

    private fun sanitizeVoiceResponse(raw: String): String {
        if (raw.isBlank()) return ""
        var cleaned = raw
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .replace("\\n", "\n")
            .trim()

        cleaned = cleaned.replaceFirst(Regex("^(assistant|system|user)\\s*:\\s*", RegexOption.IGNORE_CASE), "")
        cleaned = cleaned.replace(Regex("\n{3,}"), "\n\n").trim()

        return cleaned
    }

    private fun isPromptEcho(text: String): Boolean {
        val t = text.lowercase().trim()
        if (t.isBlank()) return true
        if (t == "assistant" || t == "user" || t == "system") return true
        if (t.startsWith("listen to the audio")) return true
        if (t.startsWith("the user just spoke in audio")) return true
        return false
    }

    private suspend fun readAudioBytes(uri: Uri): ByteArray? {
        val app = getApplication<Application>()
        return AudioConversionUtils.convertUriToFloat32Wav(app, uri)
    }
}