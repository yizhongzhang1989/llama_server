# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from logging import getLogger
from typing import List

from sentencepiece import SentencePieceProcessor


logger = getLogger()


class Tokenizer:
    """tokenizing and encoding/decoding text using SentencePiece."""
    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a SentencePiece model.

        Args:
            model_path (str): The path to the SentencePiece model file.
        """
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.

        Returns:
            List[int]: A list of token IDs.
        """
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        return self.sp_model.decode(t)


class StreamDecoder:
    def __init__(self, tokenizer: Tokenizer, new_str_callback=None):
        """
        Initialize the Tokenizer object.

        Args:
            tokenizer (Tokenizer): The tokenizer object to be used.
            new_str_callback (callable, optional): Callback function to be called when a new string is encountered.
        """
        self.tokenizer = tokenizer
        self.new_str_callback = new_str_callback

        self.token_buffer = []
        self.string_buffer = ''

    def decode_token(self, token):
        """
        Decodes a token into string_buffer as soon as possible.

        For ascii tokens, the token is decoded immediately.
        For non-ascii tokens, the token is buffered until a complete word is decoded.

        Args:
            token (int): The token to be decoded.

        Returns:
            None
        """
        piece = self.tokenizer.sp_model.IdToPiece(token)

        if piece.startswith('▁'): 
            # the piece is a new word, print the buffer
            if len(self.token_buffer) > 0:
                self.flush_token_buffer()
            # record this token
            self.token_buffer.append(token)
        else:
            # check whether the decoded text is ascii
            text = self.tokenizer.sp_model.Decode(token)

            if text.isascii():
                # the piece is a part of the previous word, append token to buffer
                self.token_buffer.append(token)
            elif text == piece:
                # the piece is not ascii, and the decoded text is a complete word, print the buffer
                if len(self.token_buffer) > 0:
                    self.flush_token_buffer()
                # record this token
                self.token_buffer.append(token)
            else:
                # the piece is not ascii, and the decoded text is not a complete word, append token to buffer
                self.token_buffer.append(token)

        # if token is EOS, print the token buffer
        if token == self.tokenizer.eos_id:
            self.flush_token_buffer()


    def flush_token_buffer(self):
        """
        Flushes the token buffer and appends the decoded tokens to the string buffer.
        """
        if len(self.token_buffer) == 0:
            return

        if self.tokenizer.sp_model.IdToPiece(self.token_buffer[0]).startswith('▁'): 
            self.string_buffer += ' '

        self.string_buffer += self.tokenizer.sp_model.Decode(self.token_buffer)
        self.token_buffer = []

        if self.new_str_callback is not None:
            self.new_str_callback(self.string_buffer)
            self.string_buffer = ''


