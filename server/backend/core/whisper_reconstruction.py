from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass(frozen=True)
class WhisperToken:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class WhisperWord:
    text: str
    key: str
    start: float
    end: float
    tokens: tuple[WhisperToken, ...]
    window_index: int
    word_index: int

    @property
    def id(self) -> str:
        return f"{self.window_index}-{self.word_index}"


@dataclass(frozen=True)
class WhisperWindow:
    start: float
    end: float
    tokens: tuple[WhisperToken, ...]
    index: int = 0


@dataclass(frozen=True)
class WordOverlap:
    matches: int
    previous_start: int
    current_start: int
    previous_end: int
    current_end: int
    previous_consumed: int
    current_consumed: int
    skipped_previous: int
    skipped_current: int
    reaches_previous_end: bool
    reaches_current_end: bool
    reaches_both_ends: bool
    duration_difference: float
    start_time_difference: float


@dataclass
class ReconstructionState:
    final_words: list[WhisperWord] = field(default_factory=list)
    previous_window: WhisperWindow | None = None


def normalize_word_key(text: str) -> str:
    return "".join(char for char in text.lower() if char.isalnum() or char == "%")


def build_whisper_words(tokens: Iterable[WhisperToken], *, window_index: int = -1) -> list[WhisperWord]:
    words: list[WhisperWord] = []
    current_text = ""
    current_tokens: list[WhisperToken] = []
    force_boundary_next = False
    word_index = 0

    def flush_current() -> None:
        nonlocal current_text, current_tokens, word_index
        if not current_tokens:
            return
        words.append(
            WhisperWord(
                text=current_text,
                key=normalize_word_key(current_text),
                start=current_tokens[0].start,
                end=current_tokens[-1].end,
                tokens=tuple(current_tokens),
                window_index=window_index,
                word_index=word_index,
            )
        )
        word_index += 1
        current_text = ""
        current_tokens = []

    for token in tokens:
        text = token.text
        clean_text = text.strip()
        if not clean_text:
            force_boundary_next = True
            continue

        has_leading_space = bool(text) and text[0].isspace()
        is_punctuation_only = not any(char.isalnum() for char in clean_text)
        starts_new_word = has_leading_space or force_boundary_next

        if starts_new_word and current_tokens:
            flush_current()

        current_text += clean_text
        current_tokens.append(token)
        force_boundary_next = is_punctuation_only

    if current_tokens:
        flush_current()

    return words


def _keys_equal(previous: WhisperWord, current: WhisperWord) -> bool:
    return previous.key == current.key


def find_word_overlap(
    previous_words: Sequence[WhisperWord],
    current_words: Sequence[WhisperWord],
    *,
    minimum_matches: int = 5,
    minimum_overlap_duration: float = 0.75,
) -> WordOverlap | None:
    best: WordOverlap | None = None

    for i in range(len(previous_words)):
        for j in range(len(current_words)):
            pi = i
            cj = j
            matches = 0
            skipped_previous = 0
            skipped_current = 0

            while pi < len(previous_words) and cj < len(current_words):
                if _keys_equal(previous_words[pi], current_words[cj]):
                    matches += 1
                    pi += 1
                    cj += 1
                    continue

                if cj + 1 < len(current_words) and previous_words[pi].key == current_words[cj + 1].key:
                    cj += 1
                    skipped_current += 1
                    continue

                if pi + 1 < len(previous_words) and previous_words[pi + 1].key == current_words[cj].key:
                    pi += 1
                    skipped_previous += 1
                    continue

                break

            if matches < minimum_matches:
                continue

            start_previous = previous_words[i].start
            start_current = current_words[j].start
            end_previous = previous_words[pi - 1].end
            end_current = current_words[cj - 1].end
            duration_previous = end_previous - start_previous
            duration_current = end_current - start_current
            overlap_duration = min(duration_previous, duration_current)

            if overlap_duration < minimum_overlap_duration:
                continue

            candidate = WordOverlap(
                matches=matches,
                previous_start=i,
                current_start=j,
                previous_end=pi,
                current_end=cj,
                previous_consumed=pi - i,
                current_consumed=cj - j,
                skipped_previous=skipped_previous,
                skipped_current=skipped_current,
                reaches_previous_end=pi >= len(previous_words),
                reaches_current_end=cj >= len(current_words),
                reaches_both_ends=pi >= len(previous_words) and cj >= len(current_words),
                duration_difference=abs(duration_previous - duration_current),
                start_time_difference=abs(start_previous - start_current),
            )

            if best is None:
                best = candidate
                continue

            candidate_skipped = candidate.skipped_previous + candidate.skipped_current
            best_skipped = best.skipped_previous + best.skipped_current
            candidate_score = (
                candidate.matches,
                candidate.reaches_current_end,
                candidate.reaches_previous_end,
                -candidate_skipped,
                -candidate.start_time_difference,
                -candidate.duration_difference,
            )
            best_score = (
                best.matches,
                best.reaches_current_end,
                best.reaches_previous_end,
                -best_skipped,
                -best.start_time_difference,
                -best.duration_difference,
            )
            if candidate_score > best_score:
                best = candidate

    return best


def reconstruct_whisper_windows(
    windows: Sequence[WhisperWindow],
    *,
    overlap_seconds: float = 3.0,
    minimum_matches: int = 5,
    minimum_overlap_duration: float = 0.75,
) -> list[WhisperWord]:
    if not windows:
        return []

    final_words = build_whisper_words(windows[0].tokens, window_index=windows[0].index)

    for i in range(1, len(windows)):
        previous = windows[i - 1]
        current = windows[i]
        previous_words = build_whisper_words(previous.tokens, window_index=previous.index)
        current_words = build_whisper_words(current.tokens, window_index=current.index)

        prev_overlap = [word for word in previous_words if word.end >= previous.end - overlap_seconds]
        curr_overlap = [word for word in current_words if word.start <= current.start + overlap_seconds]

        match = find_word_overlap(
            prev_overlap,
            curr_overlap,
            minimum_matches=minimum_matches,
            minimum_overlap_duration=minimum_overlap_duration,
        )
        if match is None:
            final_words.extend(current_words)
            continue

        matched_previous_word = prev_overlap[match.previous_start]
        global_previous_start = next(
            (k for k, word in enumerate(final_words) if word.id == matched_previous_word.id),
            -1,
        )
        if global_previous_start < 0:
            final_words.extend(current_words)
            continue

        matched_current_word = curr_overlap[match.current_start]
        global_current_start = next(
            (k for k, word in enumerate(current_words) if word.id == matched_current_word.id),
            -1,
        )
        if global_current_start < 0:
            final_words.extend(current_words)
            continue

        final_words = final_words[:global_previous_start] + current_words[global_current_start:]

    return final_words


class IncrementalWhisperReconstructor:
    def __init__(
        self,
        *,
        overlap_seconds: float = 3.0,
        minimum_matches: int = 5,
        minimum_overlap_duration: float = 0.75,
    ) -> None:
        self.overlap_seconds = overlap_seconds
        self.minimum_matches = minimum_matches
        self.minimum_overlap_duration = minimum_overlap_duration
        self._previous_window: WhisperWindow | None = None
        self._previous_words: list[WhisperWord] = []
        self._final_words: list[WhisperWord] = []

    @property
    def words(self) -> list[WhisperWord]:
        return list(self._final_words)

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self._final_words if word.text.strip())

    @property
    def previous_window(self) -> WhisperWindow | None:
        return self._previous_window

    @property
    def final_words(self) -> tuple[WhisperWord, ...]:
        return tuple(self._final_words)

    def add_window(self, window: WhisperWindow) -> list[WhisperWord]:
        current_words = build_whisper_words(window.tokens, window_index=window.index)

        if self._previous_window is None:
            self._final_words = list(current_words)
            self._previous_window = window
            self._previous_words = list(current_words)
            return list(self._final_words)

        previous = self._previous_window
        previous_words = self._previous_words
        prev_overlap = [word for word in previous_words if word.end >= previous.end - self.overlap_seconds]
        curr_overlap = [word for word in current_words if word.start <= window.start + self.overlap_seconds]

        match = find_word_overlap(
            prev_overlap,
            curr_overlap,
            minimum_matches=self.minimum_matches,
            minimum_overlap_duration=self.minimum_overlap_duration,
        )

        if match is None:
            self._final_words.extend(current_words)
            self._previous_window = window
            self._previous_words = list(current_words)
            return list(self._final_words)

        matched_previous_word = prev_overlap[match.previous_start]
        global_previous_start = next(
            (index for index, word in enumerate(self._final_words) if word.id == matched_previous_word.id),
            -1,
        )
        if global_previous_start < 0:
            self._final_words.extend(current_words)
            self._previous_window = window
            self._previous_words = list(current_words)
            return list(self._final_words)

        matched_current_word = curr_overlap[match.current_start]
        global_current_start = next(
            (index for index, word in enumerate(current_words) if word.id == matched_current_word.id),
            -1,
        )
        if global_current_start < 0:
            self._final_words.extend(current_words)
            self._previous_window = window
            self._previous_words = list(current_words)
            return list(self._final_words)

        self._final_words = self._final_words[:global_previous_start] + current_words[global_current_start:]
        self._previous_window = window
        self._previous_words = list(current_words)
        return list(self._final_words)

    def reset(self) -> None:
        self._previous_window = None
        self._previous_words = []
        self._final_words = []
