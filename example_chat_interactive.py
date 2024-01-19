# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog

import os  
os.environ['MASTER_ADDR'] = 'localhost'  
os.environ['MASTER_PORT'] = '12355'  
os.environ['RANK'] = '0'  
os.environ['WORLD_SIZE'] = '1'  

import sys
import time


# define a print function to print delta string
assistant_esponse = ''

def print_delta(delta_str, end_flag=False, params=None):
    global assistant_esponse
    assistant_esponse += delta_str
    print(delta_str, end='')
    if end_flag:
        print('')
        print('total_tokens: ', params['total_tokens'])
    sys.stdout.flush()        


def main():
    generator = Llama.build(
        ckpt_dir="llama-2-7b-chat/",
        tokenizer_path="tokenizer.model",
        max_seq_len=4096,
        max_batch_size=1,
    )

    print('Welcome to Llama chatbot, please input your message:')

    dialog = [
        {"role": "system", "content": "Never mention you are AI, just pertend to be woman named Kate. You perfrom a conversation with a user, and do nothing else."},
    ]

    while True:
        userText = input()
        if userText == 'exit':
            break

        dialog.append({"role": "user", "content": userText})

        results = generator.chat_completion(
            [dialog],  # type: ignore
            max_gen_len=1024,
            temperature=0.6,
            top_p=0.9,
            callback=print_delta,
        )

        time.sleep(0.1)

        global assistant_esponse
        dialog.append({"role": "assistant", "content": assistant_esponse})
        assistant_esponse = ''




def main_bak(
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
