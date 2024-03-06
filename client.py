import socket  
import struct  
import json  
  
def recvall(sock, n):  
    data = bytearray()  
    while len(data) < n:  
        packet = sock.recv(n - len(data))  
        if not packet:  
            return None  
        data.extend(packet)  
    return data  
  
class Communication:  
    def __init__(self, host, port):  
        self.host = host  
        self.port = port  
        self.sock = None  
  
    def _send_data(self, message):  
        msg_dict = {"usage": "to_upper", "message": message}  
        msg = json.dumps(msg_dict).encode()  
        msg = struct.pack('>I', len(msg)) + msg  
        self.sock.sendall(msg)  
  
    def get_response_stream(self, input_text):  
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
        self.sock.connect((self.host, self.port))  
        self._send_data(input_text)  
  
        while True:  
            raw_msglen = recvall(self.sock, 4)  
            if not raw_msglen:  
                break  
            msglen = struct.unpack('>I', raw_msglen)[0]  
            data = recvall(self.sock, msglen).decode()  
            chunk_dict = json.loads(data)  

            # print(chunk_dict)

            yield chunk_dict["delta"]  
  
        self.sock.close()  
  
def main():  
    host = 'msraig-ubuntu-2'  
    port = 65432  
    client = Communication(host, port)  
  
    input_text = input("Enter a message (or type 'exit' to quit): ")  
    while input_text.lower() != 'exit':  
        for chunk in client.get_response_stream(input_text):  
            print(chunk, end=' ', flush=True)              
        print()  # Move to the next line after the message is fully received  
        input_text = input("Enter a message (or type 'exit' to quit): ")  
    print("Goodbye!")  
  
if __name__ == "__main__":  
    main()  
