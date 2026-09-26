"""Incremental sliding-window transcription for Live Mode."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from core.whisper_reconstruction import (
    IncrementalWhisperReconstructor,
    WhisperToken,
    WhisperWindow,
)


@dataclass(slots=True)
class LiveStreamingConfig:
    sample_rate: int = 16000
    window_seconds: float = 2.0
    stride_seconds: float = 0.6
    min_audio_seconds: float = 0.30
    max_buffer_seconds: float = 30.0
    beam_size: int = 5
    language: str | None = None
    task: str = "transcribe"


class LiveStreamingTranscriber:
    """Low-latency sliding-window transcriber."""

    def __init__(
        self,
        *,
        backend: Any,
        backend_lock: threading.Lock,
        config: LiveStreamingConfig | None = None,
        on_partial: Callable[[str], None] | None = None,
    ) -> None:
        self._backend = backend
        self._backend_lock = backend_lock
        self._config = config or LiveStreamingConfig()
        self._on_partial = on_partial

        self._audio_lock = threading.Lock()
        self._audio = np.empty(0, dtype=np.float32)

        self._reconstructor = IncrementalWhisperReconstructor()

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._buffer_origin_seconds = 0.0
        self._window_index = 0
        self._inference_count = 0
        self._skipped_busy_count = 0

    @property
    def text(self) -> str:
        return self._reconstructor.text

    @property
    def words(self) -> list[Any]:
        return self._reconstructor.words

    @property
    def inference_count(self) -> int:
        return self._inference_count

    @property
    def skipped_busy_count(self) -> int:
        return self._skipped_busy_count

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="live-streaming-transcriber",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, wait: bool = True) -> None:
        self._stop_event.set()

        thread = self._thread
        if wait and thread is not None:
            thread.join(timeout=5.0)

        self._thread = None

    def reset(self) -> None:
        with self._audio_lock:
            self._audio = np.empty(0, dtype=np.float32)

        self._reconstructor.reset()
        self._buffer_origin_seconds = 0.0
        self._window_index = 0
        self._inference_count = 0
        self._skipped_busy_count = 0

    def feed_audio(self, pcm_data: bytes) -> None:
        if not pcm_data:
            return

        samples = np.frombuffer(pcm_data, dtype=np.int16)

        if samples.size == 0:
            return

        audio = samples.astype(np.float32) / 32768.0

        with self._audio_lock:
            self._audio = np.concatenate((self._audio, audio))

            max_samples = int(
                self._config.max_buffer_seconds * self._config.sample_rate
            )

            if len(self._audio) > max_samples:
                excess = len(self._audio) - max_samples
                self._audio = self._audio[excess:]
                self._buffer_origin_seconds += (
                    excess / self._config.sample_rate
                )

    def _worker_loop(self) -> None:
        next_run = time.monotonic()

        while not self._stop_event.is_set():
            now = time.monotonic()

            if now < next_run:
                self._stop_event.wait(min(next_run - now, 0.1))
                continue

            next_run = now + self._config.stride_seconds

            try:
                self._run_cycle()
            except Exception:
                # Partial transcription must never terminate Live Mode.
                continue

    def _run_cycle(self) -> None:
        with self._audio_lock:
            minimum_samples = int(
                self._config.min_audio_seconds * self._config.sample_rate
            )

            if len(self._audio) < minimum_samples:
                return

            window_samples = min(
                len(self._audio),
                int(
                    self._config.window_seconds
                    * self._config.sample_rate
                ),
            )

            audio = self._audio[-window_samples:].copy()

            absolute_start = (
                self._buffer_origin_seconds
                + (len(self._audio) - window_samples)
                / self._config.sample_rate
            )

            absolute_end = (
                absolute_start
                + len(audio) / self._config.sample_rate
            )

        # Do not queue stale inference work. If the normal Live Mode
        # transcription is using Whisper, this cycle is discarded.
        if not self._backend_lock.acquire(blocking=False):
            self._skipped_busy_count += 1
            return

        try:
            segments, _info = self._backend.transcribe(
                audio,
                audio_sample_rate=self._config.sample_rate,
                language=self._config.language,
                task=self._config.task,
                beam_size=self._config.beam_size,
                word_timestamps=True,
            )
        finally:
            self._backend_lock.release()

        raw_words: list[dict[str, Any]] = []

        for segment in segments:
            for word in getattr(segment, "words", []) or []:
                if not isinstance(word, dict):
                    continue

                text = word.get("word")
                start = word.get("start")
                end = word.get("end")

                if not text or start is None or end is None:
                    continue

                raw_words.append(
                    {
                        "word": str(text),
                        "start": float(start) + absolute_start,
                        "end": float(end) + absolute_start,
                        "probability": float(
                            word.get("probability", 0.0)
                        ),
                    }
                )

        if not raw_words:
            return

        whisper_words = [
            WhisperToken(
                text=word["word"],
                start=word["start"],
                end=word["end"],
            )
            for word in raw_words
        ]

        if not whisper_words:
            return

        window = WhisperWindow(
            start=absolute_start,
            end=absolute_end,
            tokens=tuple(whisper_words),
            index=self._window_index,
        )
        self._window_index += 1

        self._reconstructor.add_window(window)
        self._inference_count += 1

        if self._on_partial is not None:
            self._on_partial(self._reconstructor.text)
