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
from example_chat_interactive import get_rank_pid

client_message = None

assistant_mutex = threading.Lock()
assistant_response = ''
assistant_response_tokens = 0
assistant_end_flag = False

def recvall(sock, n):  
    # Helper function to receive exactly n bytes or return None if EOF is hit  
    data = b''  
    while len(data) < n:  
        packet = sock.recv(n - len(data))  
        if not packet:  
            return None  # Connection closed, return what has been received so far  
        data += packet  
    return data  


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
        raw_msglen = recvall(clientsocket, 4)  
        if not raw_msglen:  
            return None  # Connection closed or error  
        msglen = struct.unpack('>I', raw_msglen)[0]  

        data = recvall(clientsocket, msglen)  
        if data is None:  
            return None  # Connection closed or error during receiving the data  
        try:  
            message_dict = json.loads(data.decode())  
        except json.JSONDecodeError as e:  
            # Handle JSON decode error, e.g., log the error and/or send an error response to the client  
            return None  

        message = message_dict.get("message", "")  

        global client_message
        global assistant_response
        global assistant_response_tokens
        global assistant_end_flag

        client_message = message
        with assistant_mutex:
            assistant_response = ''
            assistant_response_tokens = 0
            assistant_end_flag = False

        for i in range(0, len(pid_list)):
            os.kill(pid_list[i], signal.SIGUSR1)

        while True:
            with assistant_mutex:
                response = assistant_response
                response_tokens = assistant_response_tokens
                end_flag = assistant_end_flag

                assistant_response = ''
                assistant_response_tokens = 0
            
            # if not ended, but no response, wait for a while
            if not end_flag and response == '':
                time.sleep(0.01)
                continue

            response_dict = {
                "message": {
                    "role": "assistant",
                    "content": response,
                },
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": response_tokens,
                    "total_tokens": 0,
                }
            }

            response_str = json.dumps(response_dict)
            send_message(clientsocket, response_str)

            if end_flag:
                break
            
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
        data = torch.empty(max(1024, length.item()), dtype=torch.uint8)

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

    while True:
        # wait for user input
        userText = wait_for_input(pid_list)  

        data = json.loads(userText)

        messages = data["messages"]
        temperature = data["temperature"]
        max_tokens = data["max_tokens"]
        top_p = data["top_p"]
        stream = data["stream"]

        results = generator.chat_completion(
            [messages],  # type: ignore
            max_gen_len=max_tokens,
            temperature=temperature,
            top_p=top_p,
            callback=print_delta if stream else None,
        )

        if not stream:
            with assistant_mutex:
                global assistant_response
                global assistant_response_tokens
                global assistant_end_flag
                assistant_response = results[0]['generation']['content']
                assistant_response_tokens = 0
                assistant_end_flag = True

if __name__ == "__main__":  
    fire.Fire(main)

