import socket  
import threading  
import struct  
import time  
import json  
import sys
import torch.distributed as dist
  
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
  
def handle_client(clientsocket, address):  
    print(f"Connection from {address} has been established.")  
    try:  
        raw_msglen = clientsocket.recv(4)  
        if not raw_msglen:  
            return  
        msglen = struct.unpack('>I', raw_msglen)[0]  
        data = clientsocket.recv(msglen).decode()  
        message_dict = json.loads(data)  
        message = message_dict.get("message", "")  
  
        msg_proc = MessageProcessor()  
  
        for chunk in msg_proc.process(message):  
            print(chunk)
            send_message(clientsocket, chunk)  
    except Exception as e:  
        print(f"Error handling client {address}: {e}")  
    finally:  
        clientsocket.close()  
        print(f"Connection with {address} closed.")  
  
def start_server(host='0.0.0.0', port=65432):  
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as serversocket:  
        serversocket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  
        serversocket.bind((host, port))  
        serversocket.listen()  
        print(f"Server listening on {host}:{port}")  
        while True:  
            clientsocket, address = serversocket.accept()  
            threading.Thread(target=handle_client, args=(clientsocket, address)).start()  


def init_distributed(port=65432):  
    # if windows
    if sys.platform == 'win32':
        dist.init_process_group(backend='gloo')  # 'gloo' or 'nccl' based on your environment    
    else:
        dist.init_process_group(backend='nccl')  # 'gloo' or 'nccl' based on your environment  
  
    rank = dist.get_rank()  
    world_size = dist.get_world_size()  
  
    print(f"Rank {rank}/{world_size} process initialized.")  
  
    # Start the server in the main process (rank 0 by convention)  
    if rank == 0:  
        print("Starting server on Rank 0.")  
        start_server(host='0.0.0.0', port=port)  
    else:  
        # Other processes could perform different tasks or assist in handling clients  
        print(f"Process {rank} ready for assisting.")  

if __name__ == "__main__":  
    init_distributed(port=65432)  
