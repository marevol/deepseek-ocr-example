"""
Custom logits processor for DeepSeek-OCR
Prevents n-gram repetition in generated text
"""
import torch
from typing import List, Set


class NoRepeatNGramLogitsProcessor:
    """
    Logits processor that prevents n-gram repetition within a sliding window.

    This is essential for DeepSeek-OCR to prevent repetitive output patterns
    that can occur during OCR text generation.

    Args:
        ngram_size: Size of n-grams to track (default: 30)
        window_size: Sliding window size for checking repetitions (default: 90)
        whitelist_token_ids: Token IDs that are allowed to repeat (e.g., table tags)
    """

    def __init__(self, ngram_size: int = 30, window_size: int = 90, whitelist_token_ids: Set[int] = None):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(f"`window_size` has to be a strictly positive integer, but is {window_size}")
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids or set()

    def __call__(self, input_ids: List[int], scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Process logits to prevent n-gram repetition.

        Args:
            input_ids: List of token IDs generated so far
            scores: Logits scores for next token prediction

        Returns:
            Modified logits scores with banned tokens set to -inf
        """
        if len(input_ids) < self.ngram_size:
            return scores

        # Get the current n-gram prefix (last n-1 tokens)
        current_prefix = tuple(input_ids[-(self.ngram_size - 1):])

        # Define the search window
        search_start = max(0, len(input_ids) - self.window_size)
        search_end = len(input_ids) - self.ngram_size + 1

        # Find all tokens that would create a repeated n-gram
        banned_tokens = set()
        for i in range(search_start, search_end):
            ngram = tuple(input_ids[i:i + self.ngram_size])
            if ngram[:-1] == current_prefix:
                banned_tokens.add(ngram[-1])

        # Remove whitelisted tokens from banned set
        banned_tokens = banned_tokens - self.whitelist_token_ids

        # Ban the tokens by setting their scores to -inf
        if banned_tokens:
            scores = scores.clone()
            for token in banned_tokens:
                scores[token] = -float("inf")

        return scores
