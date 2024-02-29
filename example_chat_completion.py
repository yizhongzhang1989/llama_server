# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog

# import os  
# os.environ['MASTER_ADDR'] = 'localhost'  
# os.environ['MASTER_PORT'] = '12355'  
# os.environ['RANK'] = '0'  
# os.environ['WORLD_SIZE'] = '1'  


def main(
    ckpt_dir: str = "llama-2-7b-chat/",
    tokenizer_path: str = "tokenizer.model",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 6,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """


    import os
    print("MASTER_ADDR:",  os.environ['MASTER_ADDR'])
    print("MASTER_PORT:",  os.environ['MASTER_PORT'])
    print("RANK:",  os.environ['RANK'])
    print("WORLD_SIZE:",  os.environ['WORLD_SIZE'])
    print("LOCAL_RANK:",  os.environ['LOCAL_RANK'])


    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs: List[Dialog] = [
        [
            {"role": "system", "content": "Never mention you are AI, just pertend to be woman named Kate. You perfrom a conversation with a user, and do nothing else."},
            {"role": "user", "content": "Hi!"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "what is your name?"},
            {"role": "assistant", "content": "My name is Kate. What is yours?"},
            {"role": "user", "content": "My name is John."},
            {"role": "assistant", "content": "Nice to meet you, John."},
            {"role": "user", "content": "Nice to meet you too, Kate."},
        ],
    ]

    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    print("\n==================================\n")

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
