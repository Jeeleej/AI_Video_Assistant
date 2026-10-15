import os
from typing import Any

import yt_dlp
from pydub import AudioSegment
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def clean_youtube_url(url: str) -> str:
    """Removes timestamp parameters (&t=...) so yt-dlp fetches cleanly."""
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    query_params.pop("t", None)
    clean_query = urlencode(query_params, doseq=True)
    return urlunparse(parsed._replace(query=clean_query))


def download_youtube_audio(url: str) -> str:
    """Downloads audio from a YouTube video and extracts it as WAV."""
    url = clean_youtube_url(url)
    output_tmpl = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")

    # Typed as Any: yt-dlp's stubs expect a TypedDict, but a plain dict works at runtime
    ydl_opts: Any = {
        'format': 'bestaudio/best',
        'outtmpl': 'downloads/%(id)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        # --- Add these options to bypass HTTP 403 ---
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web']
            }
        },
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        },
        'nocheckcertificate': True,
        'quiet': False,
        'no_warnings': False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        raw_filename = ydl.prepare_filename(info)
        # FFmpegExtractAudio replaces the extension with .wav
        wav_filename = os.path.splitext(raw_filename)[0] + ".wav"

    if not os.path.exists(wav_filename):
        raise FileNotFoundError(f"Expected downloaded audio at {wav_filename}")

    return wav_filename


def normalize_to_wav(input_path: str, target_sr: int = 16000) -> str:
    """
    Normalizes any audio/video file to 16kHz mono WAV,
    the optimal format for Whisper.
    """
    base, _ = os.path.splitext(input_path)
    output_path = f"{base}_16k.wav"

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(target_sr)
    audio.export(output_path, format="wav")

    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list[str]:
    """Splits an audio file into chunks of specified duration."""
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    base_name, _ = os.path.splitext(wav_path)

    # If audio is shorter than chunk size, return as single chunk
    if len(audio) <= chunk_ms:
        return [wav_path]

    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{base_name}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


def process_input(source: str, chunk_minutes: int = 10) -> list[str]:
    """Unified entry point for YouTube URLs or local audio/video files."""
    source = source.strip().strip('"').strip("'")

    if source.startswith(("http://", "https://")):
        print("Detected URL. Downloading audio via yt-dlp...")
        raw_audio_path = download_youtube_audio(source)
    else:
        print("Detected local file.")
        if not os.path.exists(source):
            raise FileNotFoundError(f"File not found: {source}")
        raw_audio_path = source

    print("Normalizing audio (16kHz mono)...")
    normalized_path = normalize_to_wav(raw_audio_path)

    print("Chunking audio...")
    chunks = chunk_audio(normalized_path, chunk_minutes=chunk_minutes)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks
