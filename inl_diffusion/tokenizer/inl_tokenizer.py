"""
INL Tokenizer for Text-to-Image - Wrapper around INL-Token

Uses the INL iterative tokenizer for text encoding in diffusion models.
Falls back to a simple tokenizer if INL-Token is not installed.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Union

import torch


class INLTokenizer:
    """
    INL Tokenizer for text-to-image generation.

    Wraps the INL iterative tokenizer from inl-token package.
    Provides encode/decode methods compatible with diffusion training.
    """

    def __init__(
        self,
        tokenizer_path: Optional[str] = None,
        vocab_size: int = 100000,
        max_length: int = 77,
        pad_token: str = "<|pad|>",
        unk_token: str = "<|unk|>",
        eos_token: str = "<|endoftext|>",
    ):
        """
        Initialize INL Tokenizer.

        Args:
            tokenizer_path: Path to trained tokenizer.json (from inl-token)
            vocab_size: Vocabulary size (default 100K for INL)
            max_length: Maximum sequence length (77 for CLIP compatibility)
            pad_token: Padding token
            unk_token: Unknown token
            eos_token: End of sequence token
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.eos_token = eos_token

        self._tokenizer = None
        self._iterative_tokenizer = None

        # Try to load INL tokenizer
        if tokenizer_path and Path(tokenizer_path).exists():
            self._load_inl_tokenizer(tokenizer_path)
        else:
            # Try default locations
            default_paths = [
                "inl_tokenizer/tokenizer.json",
                "../inl-token/inl_tokenizer/tokenizer.json",
                os.path.expanduser("~/.cache/inl-token/tokenizer.json"),
            ]
            for path in default_paths:
                if Path(path).exists():
                    self._load_inl_tokenizer(path)
                    break

        if self._tokenizer is None:
            print("Warning: No trained INL tokenizer found. Using fallback tokenizer.")
            print("To train the tokenizer, run: python train_inl_tokenizer.py in inl-token/")
            self._init_fallback_tokenizer()

    def _load_inl_tokenizer(self, tokenizer_path: str):
        """Load INL iterative tokenizer."""
        try:
            from tokenizers import Tokenizer

            # Load base tokenizer
            self._tokenizer = Tokenizer.from_file(tokenizer_path)

            # Try to load iterative wrapper
            try:
                import sys
                # Add inl-token to path if not installed
                inl_token_paths = [
                    str(Path(tokenizer_path).parent.parent),
                    str(Path(__file__).parent.parent.parent.parent / "inl-token"),
                ]
                for path in inl_token_paths:
                    if path not in sys.path:
                        sys.path.insert(0, path)

                from inl_tokenizer import INLIterativeTokenizer
                self._iterative_tokenizer = INLIterativeTokenizer(self._tokenizer)
                print(f"Loaded INL iterative tokenizer from {tokenizer_path}")
            except ImportError:
                print(f"Loaded base tokenizer from {tokenizer_path} (no iterative layer)")

            # Get special token IDs
            vocab = self._tokenizer.get_vocab()
            self.pad_token_id = vocab.get(self.pad_token, 0)
            self.unk_token_id = vocab.get(self.unk_token, 1)
            self.eos_token_id = vocab.get(self.eos_token, 2)
            self.vocab_size = len(vocab)

        except Exception as e:
            print(f"Could not load INL tokenizer: {e}")
            self._tokenizer = None

    def _init_fallback_tokenizer(self):
        """Initialize fallback tokenizer when INL tokenizer not available."""
        # Simple word-level tokenizer as fallback
        self._vocab = {}
        self._inverse_vocab = {}

        # Add special tokens
        special_tokens = [self.pad_token, self.unk_token, self.eos_token]
        for i, token in enumerate(special_tokens):
            self._vocab[token] = i
            self._inverse_vocab[i] = token

        self.pad_token_id = 0
        self.unk_token_id = 1
        self.eos_token_id = 2
        self._next_id = len(special_tokens)

    def _fallback_encode(self, text: str) -> List[int]:
        """Simple word-level encoding as fallback."""
        # Simple word tokenization
        words = re.findall(r'\w+|[^\w\s]', text.lower())

        token_ids = []
        for word in words:
            if word not in self._vocab:
                if self._next_id < self.vocab_size:
                    self._vocab[word] = self._next_id
                    self._inverse_vocab[self._next_id] = word
                    self._next_id += 1
                else:
                    token_ids.append(self.unk_token_id)
                    continue
            token_ids.append(self._vocab[word])

        return token_ids

    def _fallback_decode(self, token_ids: List[int]) -> str:
        """Simple word-level decoding as fallback."""
        tokens = []
        for tid in token_ids:
            if tid in self._inverse_vocab:
                token = self._inverse_vocab[tid]
                if token not in [self.pad_token, self.eos_token]:
                    tokens.append(token)
        return " ".join(tokens)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens

        Returns:
            List of token IDs
        """
        if self._iterative_tokenizer is not None:
            # Use iterative tokenizer (3-pass refinement)
            token_ids = self._iterative_tokenizer.encode(text, add_special_tokens=add_special_tokens)
        elif self._tokenizer is not None:
            # Use base tokenizer
            encoding = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
            token_ids = encoding.ids
        else:
            # Fallback
            token_ids = self._fallback_encode(text)
            if add_special_tokens:
                token_ids = token_ids + [self.eos_token_id]

        return token_ids

    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List or tensor of token IDs

        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if self._tokenizer is not None:
            return self._tokenizer.decode(token_ids)
        else:
            return self._fallback_decode(token_ids)

    def __call__(
        self,
        texts: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: Optional[str] = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize text(s) for model input.

        Args:
            texts: Single text or list of texts
            max_length: Maximum sequence length (default: self.max_length)
            padding: Padding strategy ("max_length" or "longest")
            truncation: Whether to truncate
            return_tensors: Return type ("pt" for PyTorch)

        Returns:
            Dictionary with input_ids and attention_mask
        """
        if isinstance(texts, str):
            texts = [texts]

        max_len = max_length or self.max_length

        all_input_ids = []
        all_attention_masks = []

        for text in texts:
            # Encode
            token_ids = self.encode(text, add_special_tokens=True)

            # Truncate
            if truncation and len(token_ids) > max_len:
                token_ids = token_ids[:max_len]

            # Create attention mask
            attention_mask = [1] * len(token_ids)

            # Pad
            if padding == "max_length":
                pad_length = max_len - len(token_ids)
                token_ids = token_ids + [self.pad_token_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length

            all_input_ids.append(token_ids)
            all_attention_masks.append(attention_mask)

        # Convert to tensors
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(all_attention_masks, dtype=torch.long),
            }
        else:
            return {
                "input_ids": all_input_ids,
                "attention_mask": all_attention_masks,
            }

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if self._tokenizer is not None:
            return len(self._tokenizer.get_vocab())
        return self.vocab_size

    def get_tokenization_stats(self, text: str) -> Dict:
        """
        Get tokenization statistics (for debugging).

        Returns stats about the 3-pass iterative tokenization.
        """
        if self._iterative_tokenizer is not None:
            return self._iterative_tokenizer.get_tokenization_stats(text)
        elif self._tokenizer is not None:
            encoding = self._tokenizer.encode(text)
            return {
                "text_length": len(text),
                "token_count": len(encoding.tokens),
                "compression_ratio": len(text) / len(encoding.tokens),
                "tokens_preview": encoding.tokens[:10],
                "iterative": False,
            }
        else:
            token_ids = self.encode(text)
            return {
                "text_length": len(text),
                "token_count": len(token_ids),
                "compression_ratio": len(text) / max(len(token_ids), 1),
                "iterative": False,
                "fallback": True,
            }


def load_inl_tokenizer(tokenizer_path: Optional[str] = None) -> INLTokenizer:
    """
    Load INL tokenizer for text-to-image.

    Args:
        tokenizer_path: Path to tokenizer.json (optional)

    Returns:
        INLTokenizer instance
    """
    return INLTokenizer(tokenizer_path=tokenizer_path)
