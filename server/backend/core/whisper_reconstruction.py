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


@dataclass
class ReconstructionState:
    final_words: list[WhisperWord] = field(default_factory=list)
    previous_window: WhisperWindow | None = None


def normalize_word_key(text: str) -> str:
    """
    Exact equivalent of:

        $currentText.ToLower() -replace '[^\p{L}\p{N}%]', ''

    from the original PowerShell implementation.
    """
    return "".join(
        char
        for char in text.lower()
        if char.isalnum() or char == "%"
    )


def build_whisper_words(
    tokens: Iterable[WhisperToken],
    *,
    window_index: int = -1,
) -> list[WhisperWord]:
    """
    Python port of Build-WhisperWords.ps1.

    Important behavior:

    - whitespace-only token -> force boundary before next token
    - leading whitespace -> starts a new word
    - punctuation-only token -> remains attached to current word
    - punctuation-only token forces the NEXT token to start a new word
    """
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

        is_punctuation_only = not any(
            char.isalnum()
            for char in clean_text
        )

        starts_new_word = (
            has_leading_space or force_boundary_next
        )

        if starts_new_word and current_tokens:
            flush_current()

        current_text += clean_text
        current_tokens.append(token)

        force_boundary_next = is_punctuation_only

    if current_tokens:
        flush_current()

    return words


def _keys_equal(
    previous: WhisperWord,
    current: WhisperWord,
) -> bool:
    return previous.key == current.key


def find_word_overlap(
    previous_words: Sequence[WhisperWord],
    current_words: Sequence[WhisperWord],
    *,
    minimum_matches: int = 5,
) -> WordOverlap | None:
    """
    Direct Python port of Find-WordOverlap.ps1.

    Selection criteria, in exact order:

    1. More matches.
    2. Fewer skipped words.
    3. Lower duration difference.
    """
    best: WordOverlap | None = None

    for i in range(len(previous_words)):
        for j in range(len(current_words)):
            pi = i
            cj = j

            matches = 0
            skipped_previous = 0
            skipped_current = 0

            while (
                pi < len(previous_words)
                and cj < len(current_words)
            ):
                if _keys_equal(
                    previous_words[pi],
                    current_words[cj],
                ):
                    matches += 1
                    pi += 1
                    cj += 1
                    continue

                if (
                    cj + 1 < len(current_words)
                    and previous_words[pi].key
                    == current_words[cj + 1].key
                ):
                    cj += 1
                    skipped_current += 1
                    continue

                if (
                    pi + 1 < len(previous_words)
                    and previous_words[pi + 1].key
                    == current_words[cj].key
                ):
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

            duration_previous = (
                end_previous - start_previous
            )

            duration_current = (
                end_current - start_current
            )

            duration_difference = abs(
                duration_previous - duration_current
            )

            previous_consumed = pi - i
            current_consumed = cj - j

            reaches_previous_end = (
                pi >= len(previous_words)
            )

            reaches_current_end = (
                cj >= len(current_words)
            )

            candidate = WordOverlap(
                matches=matches,
                previous_start=i,
                current_start=j,
                previous_end=pi,
                current_end=cj,
                previous_consumed=previous_consumed,
                current_consumed=current_consumed,
                skipped_previous=skipped_previous,
                skipped_current=skipped_current,
                reaches_previous_end=reaches_previous_end,
                reaches_current_end=reaches_current_end,
                reaches_both_ends=(
                    reaches_previous_end
                    and reaches_current_end
                ),
                duration_difference=duration_difference,
            )

            if best is None:
                best = candidate
                continue

            if candidate.matches > best.matches:
                best = candidate
                continue

            if candidate.matches == best.matches:
                candidate_skipped = (
                    candidate.skipped_previous
                    + candidate.skipped_current
                )

                best_skipped = (
                    best.skipped_previous
                    + best.skipped_current
                )

                if candidate_skipped < best_skipped:
                    best = candidate
                    continue

                if (
                    candidate_skipped == best_skipped
                    and candidate.duration_difference
                    < best.duration_difference
                ):
                    best = candidate

    return best


def reconstruct_whisper_windows(
    windows: Sequence[WhisperWindow],
    *,
    overlap_seconds: float = 3.0,
    minimum_matches: int = 5,
) -> list[WhisperWord]:
    """
    Direct Python port of Reconstruct-WhisperWindows.ps1.

    PreviousConsumed / CurrentConsumed are diagnostic only.
    They do NOT determine the reconstruction slicing.
    """
    if not windows:
        return []

    final_words = build_whisper_words(
        windows[0].tokens,
        window_index=0,
    )

    for i in range(1, len(windows)):
        previous = windows[i - 1]
        current = windows[i]

        previous_words = build_whisper_words(
            previous.tokens,
            window_index=i - 1,
        )

        current_words = build_whisper_words(
            current.tokens,
            window_index=i,
        )

        prev_overlap = [
            word
            for word in previous_words
            if word.end >= previous.end - overlap_seconds
        ]

        curr_overlap = [
            word
            for word in current_words
            if word.start <= current.start + overlap_seconds
        ]

        match = find_word_overlap(
            prev_overlap,
            curr_overlap,
            minimum_matches=minimum_matches,
        )

        if match is None:
            final_words.extend(current_words)
            continue

        if not (
            0 <= match.previous_start < len(prev_overlap)
        ):
            continue

        matched_previous_word = (
            prev_overlap[match.previous_start]
        )

        global_previous_start = -1

        for k, word in enumerate(final_words):
            if word.id == matched_previous_word.id:
                global_previous_start = k
                break

        if global_previous_start < 0:
            continue

        if not (
            0 <= match.current_start < len(curr_overlap)
        ):
            continue

        matched_current_word = (
            curr_overlap[match.current_start]
        )

        global_current_start = -1

        for k, word in enumerate(current_words):
            if word.id == matched_current_word.id:
                global_current_start = k
                break

        if global_current_start < 0:
            continue

        previous_match_end = (
            match.previous_start
            + match.previous_consumed
        )

        current_match_end = (
            match.current_start
            + match.current_consumed
        )

        previous_orphan_suffix = (
            len(prev_overlap) - previous_match_end
        )

        current_orphan_suffix = (
            len(curr_overlap) - current_match_end
        )

        _ = previous_orphan_suffix
        _ = current_orphan_suffix

        prefix = final_words[:global_previous_start]

        current_from_start = (
            current_words[global_current_start:]
        )

        if not current_from_start:
            continue

        final_words = prefix + current_from_start

    return final_words


class IncrementalWhisperReconstructor:
    """
    Stateful incremental version of Reconstruct-WhisperWindows.

    Unlike reconstruct_whisper_windows(), this class does NOT retain
    and reprocess the complete window history.

    Each new window is reconciled only against the immediately
    preceding Whisper window.
    """

    def __init__(
        self,
        *,
        overlap_seconds: float = 3.0,
        minimum_matches: int = 5,
    ) -> None:
        self.overlap_seconds = overlap_seconds
        self.minimum_matches = minimum_matches

        self._previous_window: WhisperWindow | None = None
        self._previous_words: list[WhisperWord] = []
        self._final_words: list[WhisperWord] = []

    @property
    def words(self) -> list[WhisperWord]:
        return list(self._final_words)

    @property
    def text(self) -> str:
        return " ".join(
            word.text
            for word in self._final_words
            if word.text.strip()
        )

    @property
    def previous_window(self) -> WhisperWindow | None:
        return self._previous_window

    @property
    def final_words(self) -> tuple[WhisperWord, ...]:
        return tuple(self._final_words)

    def add_window(
        self,
        window: WhisperWindow,
    ) -> list[WhisperWord]:
        """
        Add exactly one completed Whisper window.

        The first window initializes the transcript.

        Every subsequent window is reconciled against the immediately
        preceding window using the same overlap algorithm as the
        original PowerShell implementation.
        """
        current_words = build_whisper_words(
            window.tokens,
            window_index=window.index,
        )

        if self._previous_window is None:
            self._final_words = list(current_words)
            self._previous_window = window
            self._previous_words = list(current_words)
            return list(self._final_words)

        previous = self._previous_window
        previous_words = self._previous_words

        prev_overlap = [
            word
            for word in previous_words
            if word.end >= (
                previous.end - self.overlap_seconds
            )
        ]

        curr_overlap = [
            word
            for word in current_words
            if word.start <= (
                window.start + self.overlap_seconds
            )
        ]

        match = find_word_overlap(
            previous_words=prev_overlap,
            current_words=curr_overlap,
            minimum_matches=self.minimum_matches,
        )

        if match is None:
            self._final_words.extend(current_words)

            self._previous_window = window
            self._previous_words = list(current_words)

            return list(self._final_words)

        if not (
            0 <= match.previous_start < len(prev_overlap)
        ):
            self._previous_window = window
            self._previous_words = list(current_words)

            return list(self._final_words)

        matched_previous_word = (
            prev_overlap[match.previous_start]
        )

        global_previous_start = -1

        for index, word in enumerate(self._final_words):
            if word.id == matched_previous_word.id:
                global_previous_start = index
                break

        if global_previous_start < 0:
            self._final_words.extend(current_words)

            self._previous_window = window
            self._previous_words = list(current_words)

            return list(self._final_words)

        if not (
            0 <= match.current_start < len(curr_overlap)
        ):
            self._previous_window = window
            self._previous_words = list(current_words)

            return list(self._final_words)

        matched_current_word = (
            curr_overlap[match.current_start]
        )

        global_current_start = -1

        for index, word in enumerate(current_words):
            if word.id == matched_current_word.id:
                global_current_start = index
                break

        if global_current_start < 0:
            self._final_words.extend(current_words)

            self._previous_window = window
            self._previous_words = list(current_words)

            return list(self._final_words)

        previous_match_end = (
            match.previous_start
            + match.previous_consumed
        )

        current_match_end = (
            match.current_start
            + match.current_consumed
        )

        previous_orphan_suffix = (
            len(prev_overlap) - previous_match_end
        )

        current_orphan_suffix = (
            len(curr_overlap) - current_match_end
        )

        _ = previous_orphan_suffix
        _ = current_orphan_suffix

        prefix = self._final_words[
            :global_previous_start
        ]

        current_from_start = current_words[
            global_current_start:
        ]

        if current_from_start:
            self._final_words = (
                prefix + current_from_start
            )

        self._previous_window = window
        self._previous_words = list(current_words)

        return list(self._final_words)

    def reset(self) -> None:
        """
        Completely reset incremental reconstruction state.
        """
        self._previous_window = None
        self._previous_words = []
        self._final_words = []
