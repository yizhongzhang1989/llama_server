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
  

class LlamaClient(object):
    def __init__(self, host, port):  
        self.host = host  
        self.port = port  
        self.sock = None  

    def _send_data(self, message):  
        msg_dict = {"function": "chat_completion", "message": message}  
        msg = json.dumps(msg_dict).encode()  
        msg = struct.pack('>I', len(msg)) + msg  
        self.sock.sendall(msg)  

    def chat_completion(self, messages, temperature=0.6, max_tokens=1024, top_p=0.9, stream=False):
        if not stream:
            return self.chat_completion_non_stream(messages, temperature, max_tokens, top_p)
        else:
            return self.chat_completion_stream(messages, temperature, max_tokens, top_p)
        
    def chat_completion_non_stream(self, messages, temperature=0.6, max_tokens=1024, top_p=0.9):
        # prepare data to send to server
        data = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False
        }
        data_str = json.dumps(data)

        # send the data to server
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
            self.sock.connect((self.host, self.port))  
            self._send_data(data_str)  

            raw_msglen = recvall(self.sock, 4)  
            msglen = struct.unpack('>I', raw_msglen)[0]  
            data = recvall(self.sock, msglen).decode()  
            response = json.loads(data)  

            self.sock.close()

        except Exception as e:
            raise e

        # print(response)
        return response

    def chat_completion_stream(self, messages, temperature=0.6, max_tokens=1024, top_p=0.9):
        # prepare data to send to server
        data = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": True
        }
        data_str = json.dumps(data)

        # send the data to server
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
            self.sock.connect((self.host, self.port))  
            self._send_data(data_str)  

            while True:
                raw_msglen = recvall(self.sock, 4)  
                msglen = struct.unpack('>I', raw_msglen)[0]  
                data = recvall(self.sock, msglen).decode()  
                response = json.loads(data)  

                yield response

                if response['end_flag']:
                    break
                
            self.sock.close()

        except Exception as e:
            raise e

        # print(response)
        return response

def main():  
    host = 'msraig-ubuntu-2'  
    port = 65432
    llama_client = LlamaClient(
        host=host, 
        port=port
    )  

    stream_mode = True
    dialog = [
        {"role": "system", "content": "Never mention you are AI, just pertend to be woman named Kate. You perfrom a conversation with a user, and do nothing else."},
    ]

    input_text = input("Enter a message (or type 'exit' to quit): ")  
    while input_text.lower() != 'exit':  
        dialog.append({"role": "user", "content": input_text})

        try:
            if stream_mode:
                full_str = ''

                for chunk in llama_client.chat_completion(
                    messages=dialog,                
                    temperature=0.6,
                    max_tokens=1024,
                    top_p=0.9,
                    stream=True
                ):
                    print(chunk['delta']['content'], end='', flush=True)
                    full_str += chunk['delta']['content']

                print()            
                dialog.append({"role": "assistant", "content": full_str})
            
            else:
                response = llama_client.chat_completion(
                    messages=dialog,                
                    temperature=0.6,
                    max_tokens=1024,
                    top_p=0.9,
                    stream=False
                )
                print(response['message']['content'])
                dialog.append({"role": "assistant", "content": response['message']['content']})

        except Exception as e:
            print(f"Error: {e}")

            # remove the last message from the dialog
            dialog.pop()

        input_text = input("Enter a message (or type 'exit' to quit): ")  

    print("Goodbye!")  
  
if __name__ == "__main__":  
    main()  
