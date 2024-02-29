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

import sys
import time

import torch.distributed as dist  
import torch
import os
  
rank = int(os.environ['RANK'])


# define a print function to print delta string
stream_mode = True

assistant_esponse = ''

def print_delta(delta_str, end_flag=False, params=None):
    global assistant_esponse
    assistant_esponse += delta_str
    print(delta_str, end='')
    if end_flag:
        print('')
        print('total_tokens: ', params['total_tokens'])
    sys.stdout.flush()        


def read_and_broadcast_input():  
    if rank == 0:  
        userText = input()  
        # For safety, you might want to ensure the text is not too long,  
        # or handle that case appropriately before broadcasting.  
    else:  
        userText = None  
  
    # Assuming userText is a string, we need to convert it to a tensor for broadcasting.  
    # One way is to encode it to bytes, get the length, and then create two tensors:  
    # one for the length and one for the byte values.  
    if rank == 0:  
        userText_bytes = userText.encode()  
        length = torch.tensor([len(userText_bytes)], dtype=torch.long)  
        data = torch.tensor(list(userText_bytes), dtype=torch.uint8)  
    else:  
        length = torch.tensor([0], dtype=torch.long)  
        # Allocate tensor based on maximum expected length, or use dynamic resizing for efficiency.  
        data = torch.empty(1024, dtype=torch.uint8)  # Example size, adjust as needed.  
  
    # Broadcast the length first, then the actual data.  
    dist.broadcast(length, src=0)  
    dist.broadcast(data[:length.item()], src=0)  
  
    if rank != 0:  
        userText = bytes(data[:length.item()]).decode()  
  
    return userText  


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 6,
    max_gen_len: Optional[int] = None
):    
    generator = Llama.build(
        ckpt_dir,
        tokenizer_path,
        max_seq_len,
        max_batch_size,
    )

    print('Welcome to Llama chatbot, please input your message:')

    dialog = [
        {"role": "system", "content": "Never mention you are AI, just pertend to be woman named Kate. You perfrom a conversation with a user, and do nothing else."},
    ]

    while True:
        userText = read_and_broadcast_input()  
        # userText = input()
        if userText == 'exit':
            break

        dialog.append({"role": "user", "content": userText})

        results = generator.chat_completion(
            [dialog],  # type: ignore
            max_gen_len=1024,
            temperature=0.6,
            top_p=0.9,
            callback=print_delta if stream_mode else None,
        )

        if not stream_mode:
            print(results[0]['generation']['content'])

        time.sleep(0.1)

        global assistant_esponse
        dialog.append({"role": "assistant", "content": assistant_esponse})
        assistant_esponse = ''


if __name__ == "__main__":
    fire.Fire(main)
