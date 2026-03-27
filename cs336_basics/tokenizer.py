from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex as re

GPT2_PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _tokenize_text(text: str) -> list[str]:
    return re.findall(GPT2_PRETOKEN_PATTERN, text)


def _split_on_special_tokens(text: str, special_tokens: list[str] | None) -> list[str]:
    if not special_tokens:
        return _tokenize_text(text)

    escaped = sorted((re.escape(token) for token in special_tokens), key=len, reverse=True)
    pattern = "(" + "|".join(escaped) + ")"
    parts = re.split(pattern, text)
    special_token_set = set(special_tokens)
    output: list[str] = []

    for part in parts:
        if not part:
            continue
        if part in special_token_set:
            output.append(part)
        else:
            output.extend(_tokenize_text(part))
    return output


def _word_to_bytes(word: str) -> tuple[bytes, ...]:
    raw = word.encode("utf-8")
    return tuple(bytes([value]) for value in raw)


def _merge_pretoken(pretoken: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    merged_token = pair[0] + pair[1]
    output: list[bytes] = []
    idx = 0
    while idx < len(pretoken):
        if idx + 1 < len(pretoken) and (pretoken[idx], pretoken[idx + 1]) == pair:
            output.append(merged_token)
            idx += 2
        else:
            output.append(pretoken[idx])
            idx += 1
    return tuple(output)


def _build_base_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {}
    seen: set[bytes] = set()
    next_id = 0

    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in seen:
            vocab[next_id] = token_bytes
            seen.add(token_bytes)
            next_id += 1

    for byte_value in range(256):
        token_bytes = bytes([byte_value])
        if token_bytes not in seen:
            vocab[next_id] = token_bytes
            seen.add(token_bytes)
            next_id += 1

    return vocab


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = _build_base_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    special_token_set = set(special_tokens)
    word_counts: Counter[tuple[bytes, ...]] = Counter()
    with open(input_path, encoding="utf-8") as f:
        for chunk in f:
            for token in _split_on_special_tokens(chunk, special_tokens):
                if token in special_token_set:
                    word_counts[(token.encode("utf-8"),)] += 1
                else:
                    word_counts[_word_to_bytes(token)] += 1

    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    for pretoken, count in word_counts.items():
        for idx in range(len(pretoken) - 1):
            pair_counts[(pretoken[idx], pretoken[idx + 1])] += count

    while len(vocab) < vocab_size:

        if not pair_counts:
            break

        best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
        merged_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[len(vocab)] = merged_token

        updates: list[tuple[tuple[bytes, ...], tuple[bytes, ...], int]] = []
        for pretoken, count in tuple(word_counts.items()):
            if len(pretoken) < 2:
                continue
            if best_pair not in zip(pretoken, pretoken[1:]):
                continue
            updates.append((pretoken, _merge_pretoken(pretoken, best_pair), count))

        for old_pretoken, new_pretoken, count in updates:
            for idx in range(len(old_pretoken) - 1):
                old_pair = (old_pretoken[idx], old_pretoken[idx + 1])
                pair_counts[old_pair] -= count
                if pair_counts[old_pair] == 0:
                    del pair_counts[old_pair]

            for idx in range(len(new_pretoken) - 1):
                new_pair = (new_pretoken[idx], new_pretoken[idx + 1])
                pair_counts[new_pair] += count

            del word_counts[old_pretoken]
            word_counts[new_pretoken] += count

        pair_counts.pop(best_pair, None)

    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.byte_to_id = {value: key for key, value in self.vocab.items()}
        self.special_tokens = list(special_tokens or [])
        self.special_token_bytes = {token.encode("utf-8") for token in self.special_tokens}

        for token_bytes in self.special_token_bytes:
            if token_bytes not in self.byte_to_id:
                token_id = len(self.vocab)
                self.vocab[token_id] = token_bytes
                self.byte_to_id[token_bytes] = token_id

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {int(token_id): token.encode("utf-8") for token, token_id in raw_vocab.items()}

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split(" ")
                if len(parts) != 2:
                    continue
                merges.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))

        return cls(vocab, merges, special_tokens)

    def _apply_merges(self, pretoken: tuple[bytes, ...]) -> tuple[bytes, ...]:
        pieces = list(pretoken)
        if len(pieces) < 2:
            return tuple(pieces)

        while len(pieces) >= 2:
            candidate_pairs = [(pieces[idx], pieces[idx + 1]) for idx in range(len(pieces) - 1)]
            ranked_pairs = [pair for pair in candidate_pairs if pair in self.merge_ranks]
            if not ranked_pairs:
                break

            best_pair = min(ranked_pairs, key=self.merge_ranks.__getitem__)
            merged: list[bytes] = []
            idx = 0
            while idx < len(pieces):
                if idx + 1 < len(pieces) and (pieces[idx], pieces[idx + 1]) == best_pair:
                    merged.append(pieces[idx] + pieces[idx + 1])
                    idx += 2
                else:
                    merged.append(pieces[idx])
                    idx += 1
            pieces = merged

        return tuple(pieces)

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for token in _split_on_special_tokens(text, self.special_tokens):
            token_bytes = token.encode("utf-8")
            if token in self.special_tokens:
                token_ids.append(self.byte_to_id[token_bytes])
                continue

            for piece in self._apply_merges(_word_to_bytes(token)):
                token_ids.append(self.byte_to_id[piece])
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, token_ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in token_ids)
        return token_bytes.decode("utf-8", errors="replace")
