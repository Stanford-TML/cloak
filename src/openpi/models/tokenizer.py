import logging

import numpy as np
import sentencepiece

import openpi.shared.download as download


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            # This is the Pi05 format, where the state is part of the discrete language input.
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # This is the Pi0 format, where the state is part of the continuous action expert input.
            # tokenize "\n" separately as the "start of answer" token
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)

    def tokenize_batch(
        self, prompts: list[str], states: np.ndarray | None = None, *, debug: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batched tokenize: single SentencePiece C call for all rows."""
        cleaned = [p.strip().replace("_", " ").replace("\n", " ") for p in prompts]
        if states is not None:
            disc = np.digitize(states, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1  # (B, S)
            full = [
                f"Task: {c}, State: {' '.join(map(str, row))};\nAction: "
                for c, row in zip(cleaned, disc, strict=True)
            ]
            token_lists = self._tokenizer.encode(full, add_bos=True)
        else:
            full = [c + "\n" for c in cleaned]
            token_lists = self._tokenizer.encode(cleaned, add_bos=True)
            nl = self._tokenizer.encode("\n")
            token_lists = [tl + nl for tl in token_lists]

        if debug:
            for s in full:
                print(f"[tokenize_batch] full prompt: {s!r}")

        B = len(prompts)
        tokens = np.zeros((B, self._max_len), dtype=int)
        mask = np.zeros((B, self._max_len), dtype=np.bool_)
        overflow = 0
        for i, tl in enumerate(token_lists):
            L = min(len(tl), self._max_len)
            tokens[i, :L] = tl[:L]
            mask[i, :L] = True
            if len(tl) > self._max_len:
                overflow += 1
        if overflow:
            logging.warning(
                f"{overflow}/{B} prompts exceeded max length ({self._max_len}) and were truncated. "
                "Consider increasing the `max_token_len` in your model config if this happens frequently."
            )
        return tokens, mask
