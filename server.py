from typing import List, Optional
import fire
from llama import Llama, Dialog
import socket  
import threading  
import struct  
import time  
import json  
import sys
import os
import signal
import torch
import torch.distributed as dist
from example_chat_interactive import get_rank_pid, assistant_response

client_message = None


def print_delta(delta_str, end_flag=False, params=None):
    global assistant_response
    assistant_response += delta_str
    print(delta_str, end='')
    if end_flag:
        print('')
        print('total_tokens: ', params['total_tokens'])
    sys.stdout.flush()        

class MessageProcessor:  
    def process(self, message):  
        words = message.upper().split()  
        for word in words:  
            time.sleep(0.1)  # Introduce a 100ms delay here, in the processing  
            chunk = json.dumps({"usage": "to_upper", "delta": word})  
            yield chunk  
  
def send_message(clientsocket, message):  
    encoded_message = message.encode()  
    message_length = len(encoded_message)  
    clientsocket.sendall(struct.pack('>I', message_length) + encoded_message)  
  
def handle_client(pid_list, clientsocket, address):  
    print(f"Connection from {address} has been established.")  
    try:  
        raw_msglen = clientsocket.recv(4)  
        if not raw_msglen:  
            return  
        msglen = struct.unpack('>I', raw_msglen)[0]  
        data = clientsocket.recv(msglen).decode()  
        message_dict = json.loads(data)  
        message = message_dict.get("message", "")  

        global client_message
        client_message = message

        for i in range(0, len(pid_list)):
            os.kill(pid_list[i], signal.SIGUSR1)

        msg_proc = MessageProcessor()  
  
        for chunk in msg_proc.process(message):  
            # print(chunk)
            send_message(clientsocket, chunk)  
    except Exception as e:  
        print(f"Error handling client {address}: {e}")  
    finally:  
        clientsocket.close()  
        # print(f"Connection with {address} closed.")  
  
def start_server(pid_list, host='0.0.0.0', port=65432):  
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as serversocket:  
        serversocket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  
        serversocket.bind((host, port))  
        serversocket.listen()  
        print(f"Server listening on {host}:{port}")  
        while True:  
            clientsocket, address = serversocket.accept()  
            threading.Thread(target=handle_client, args=(pid_list, clientsocket, address)).start()  

def signal_handler(signum, frame):  
    pass

def wait_for_input(pid_list):  
    signal.signal(signal.SIGUSR1, signal_handler)
    signal.pause()

    rank = dist.get_rank()

    # boardcase data length
    if rank == 0:
        userText = client_message
        userText_bytes = userText.encode()  
        length = torch.tensor([len(userText_bytes)], dtype=torch.long)
    else:
        userText = None
        length = torch.tensor([0], dtype=torch.long)  

    dist.broadcast(length, src=0)

    # broadcast data
    if rank == 0:
        data = torch.tensor(list(userText_bytes), dtype=torch.uint8)
    else:
        data = torch.empty(min(1024, length.item()), dtype=torch.uint8)

    dist.broadcast(data[:length.item()], src=0)

    if rank != 0:  
        userText = bytes(data[:length.item()]).decode()  

    # print userText with rank
    print(f"Rank {rank} received userText: {userText}")

    return userText  


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 6,
    max_gen_len: Optional[int] = None,
    port=65432
):  
    dist.init_process_group("nccl")

    # Initialize the process group    
    rank = dist.get_rank()  
    world_size = dist.get_world_size()    
    print(f"Rank {rank}/{world_size} process started.")  

    pid_list = get_rank_pid()
    print(f"Rank {rank} received pid list: {pid_list}")
    dist.barrier()

    # Initialize the model
    generator = Llama.build(
        ckpt_dir,
        tokenizer_path,
        max_seq_len,
        max_batch_size,
    )
    dist.barrier()

    # start server in a separate thread
    if rank == 0:
        print("Starting server on Rank 0.")
        threading.Thread(target=start_server, args=(pid_list, '0.0.0.0', port)).start()
    dist.barrier()


    stream_mode = True
    dialog = [
        {"role": "system", "content": "Never mention you are AI, just pertend to be woman named Kate. You perfrom a conversation with a user, and do nothing else."},
    ]

    while True:
        userText = wait_for_input(pid_list)  
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

        global assistant_response
        dialog.append({"role": "assistant", "content": assistant_response})
        assistant_response = ''


if __name__ == "__main__":  
    fire.Fire(main)

