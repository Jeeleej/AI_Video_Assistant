import os
import whisper

# Whisper model size: tiny, base, small, medium, large, turbo.
# "base" or "small" is a good default for CPU; set WHISPER_MODEL in .env to override.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "turbo")

_model = None


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def load_model():
    """Lazily load and cache the Whisper model in memory."""
    global _model

    if _model is None:
        device = "cuda" if _has_cuda() else "cpu"
        print(f"Loading Whisper model: {WHISPER_MODEL} on {device} ...")
        _model = whisper.load_model(WHISPER_MODEL, device=device)
        print("Whisper model loaded.")
    return _model


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Transcribe an audio chunk using Whisper.
    - 'english'  -> standard speech-to-text in English
    - 'hinglish' -> transcribes and translates mixed Hindi/English speech to English
    """
    model = load_model()
    use_fp16 = _has_cuda()  # fp16 is not supported on CPU

    if language.lower() == "hinglish":
        # Whisper's translate task converts mixed Hindi/English speech to English text
        result = model.transcribe(chunk_path, task="translate", fp16=use_fp16)
    else:
        result = model.transcribe(
            chunk_path, task="transcribe", language="en", fp16=use_fp16
        )

    return str(result.get("text", "")).strip()


def transcribe_all(chunks: list, language: str = "english") -> str:
    """Process all audio chunks sequentially through Whisper."""
    full_transcript_parts = []

    print("Using Whisper for transcription.")

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language=language)
        if text:
            full_transcript_parts.append(text)

    transcript = " ".join(full_transcript_parts).strip()
    if not transcript:
        raise ValueError("Whisper returned an empty transcript. Is there speech in the audio?")

    print("Transcription complete.")

    # Free up GPU memory for downstream operations
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return transcript
