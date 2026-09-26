"""Real Whisper HTTP sliding-window reconstruction test.

Uses the real Seminar Whisper server on port 8081 against a local WAV file.
This is intentionally an offline/deterministic window scheduler: it tests
real ASR + word timestamps + incremental reconstruction without microphone
capture or scheduler timing.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
import wave
from pathlib import Path
from urllib import request


SERVER_URL = "http://127.0.0.1:8081/inference"
DEFAULT_AUDIO = Path(r"G:\Descargas\diplomado-test-20s.wav")
SAMPLE_RATE = 16000
WINDOW_SECONDS = 2.0
STRIDE_SECONDS = 0.6


def make_multipart(
    wav_bytes: bytes,
    *,
    filename: str,
    fields: dict[str, str],
) -> tuple[bytes, str]:
    boundary = "----TranscriptionSuiteSeminar" + uuid.uuid4().hex
    body = bytearray()

    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        )
        body.extend(value.encode())
        body.extend(b"\r\n")

    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        (
            f'Content-Disposition: form-data; name="file"; '
            f'filename="{filename}"\r\n'
        ).encode()
    )
    body.extend(b"Content-Type: audio/wav\r\n\r\n")
    body.extend(wav_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())

    return bytes(body), f"multipart/form-data; boundary={boundary}"


def transcribe_window(wav_bytes: bytes, filename: str) -> tuple[dict, float]:
    body, content_type = make_multipart(
        wav_bytes,
        filename=filename,
        fields={
            "response_format": "verbose_json",
            "language": "es",
            "word_timestamps": "true",
        },
    )

    req = request.Request(
        SERVER_URL,
        data=body,
        method="POST",
        headers={
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
        },
    )

    started = time.perf_counter()

    with request.urlopen(req, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))

    elapsed = time.perf_counter() - started
    return result, elapsed


def extract_words(result: dict, window_start: float) -> list[tuple[str, float, float]]:
    words: list[tuple[str, float, float]] = []

    for segment in result.get("segments", []) or []:
        for word in segment.get("words", []) or []:
            text = str(word.get("word", "")).strip()
            start = word.get("start")
            end = word.get("end")

            if not text or start is None or end is None:
                continue

            # Whisper timestamps are relative to this window.
            # Convert them to absolute source-audio time.
            words.append(
                (
                    text,
                    float(start) + window_start,
                    float(end) + window_start,
                )
            )

    return words


def main() -> int:
    audio_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else DEFAULT_AUDIO
    )

    if not audio_path.exists():
        print(f"FAIL: audio not found: {audio_path}")
        return 1

    with wave.open(str(audio_path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        rate = wav.getframerate()
        frame_count = wav.getnframes()
        pcm = wav.readframes(frame_count)

    if channels != 1 or sample_width != 2 or rate != SAMPLE_RATE:
        print(
            "FAIL: expected WAV PCM 16-bit mono 16000 Hz; "
            f"got channels={channels}, width={sample_width}, rate={rate}"
        )
        return 1

    total_seconds = frame_count / rate
    window_samples = int(WINDOW_SECONDS * rate)
    stride_samples = int(STRIDE_SECONDS * rate)

    print(f"AUDIO: {audio_path}")
    print(f"DURATION: {total_seconds:.3f}s")
    print(f"WINDOW: {WINDOW_SECONDS:.1f}s")
    print(f"STRIDE: {STRIDE_SECONDS:.1f}s")
    print(f"SERVER: {SERVER_URL}")
    print()

    from core.whisper_reconstruction import (
        IncrementalWhisperReconstructor,
        WhisperToken,
        WhisperWindow,
    )

    reconstructor = IncrementalWhisperReconstructor(
        overlap_seconds=3.0,
        minimum_matches=5,
    )

    timings: list[float] = []
    inference_count = 0
    window_start_sample = 0

    while window_start_sample < frame_count:
        window_end_sample = min(
            window_start_sample + window_samples,
            frame_count,
        )

        # Do not process a final fragment shorter than the configured
        # minimum useful window. The 20s test therefore uses full windows.
        if window_end_sample - window_start_sample < int(0.5 * rate):
            break

        window_start = window_start_sample / rate
        window_end = window_end_sample / rate

        start_byte = window_start_sample * sample_width
        end_byte = window_end_sample * sample_width
        window_pcm = pcm[start_byte:end_byte]

        # Build a valid WAV container around the PCM window.
        import io

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(SAMPLE_RATE)
            out.writeframes(window_pcm)

        wav_window = buffer.getvalue()

        result, elapsed = transcribe_window(
            wav_window,
            filename=f"window-{inference_count}.wav",
        )
        timings.append(elapsed)

        tokens = [
            WhisperToken(
                text=text,
                start=start,
                end=end,
            )
            for text, start, end in extract_words(
                result,
                window_start,
            )
        ]

        if not tokens:
            print(
                f"WINDOW {inference_count:02d} "
                f"{window_start:5.1f}-{window_end:5.1f}s "
                f"{elapsed:.3f}s -> NO WORDS"
            )
        else:
            window = WhisperWindow(
                start=window_start,
                end=window_end,
                tokens=tuple(tokens),
                index=inference_count,
            )
            reconstructor.add_window(window)

            print(
                f"WINDOW {inference_count:02d} "
                f"{window_start:5.1f}-{window_end:5.1f}s "
                f"{elapsed:.3f}s"
            )
            print(f"  PARTIAL: {reconstructor.text}")

        inference_count += 1

        if window_end_sample >= frame_count:
            break

        window_start_sample += stride_samples

    print()
    print("========================================")
    print("REAL WHISPER STREAMING TEST")
    print("========================================")
    print(f"INFERENCES: {inference_count}")
    print(f"MIN TIME: {min(timings):.3f}s")
    print(f"MAX TIME: {max(timings):.3f}s")
    print(f"AVG TIME: {sum(timings) / len(timings):.3f}s")
    print(f"FINAL TEXT: {reconstructor.text}")
    print()

    if not reconstructor.text.strip():
        print("FAIL: no reconstructed text")
        return 1

    print("REAL WHISPER STREAMING TEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
