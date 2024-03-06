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
  
class LlamaServerAPI(object):  
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
  
    def tmp(self, messages, temperature=0.6, max_tokens=1024, top_p=0.9, stream=False):
        print(messages)

        # prepare data to send to server
        data = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream
        }

        return None

        data_str = json.dumps(data)

        # send the data to server
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
        self.sock.connect((self.host, self.port))  
        self._send_data(data_str)  

        response = None

        if stream:
            while True:  
                raw_msglen = recvall(self.sock, 4)  
                if not raw_msglen:  
                    break  
                msglen = struct.unpack('>I', raw_msglen)[0]  
                data = recvall(self.sock, msglen).decode()  
                chunk_dict = json.loads(data)  

                yield chunk_dict["delta"]  
        else:
            raw_msglen = recvall(self.sock, 4)  
            msglen = struct.unpack('>I', raw_msglen)[0]  
            data = recvall(self.sock, msglen).decode()  
            response = json.loads(data)  
  
        self.sock.close()  

        return response
    

    def chat_completion(self, messages, temperature=0.6, max_tokens=1024, top_p=0.9, stream=False):
        print(messages)

        return None

        # prepare data to send to server
        data = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream
        }
        data_str = json.dumps(data)

        # send the data to server
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
        self.sock.connect((self.host, self.port))  
        self._send_data(data_str)  

        response = None

        if stream:
            while True:  
                raw_msglen = recvall(self.sock, 4)  
                if not raw_msglen:  
                    break  
                msglen = struct.unpack('>I', raw_msglen)[0]  
                data = recvall(self.sock, msglen).decode()  
                chunk_dict = json.loads(data)  

                yield chunk_dict["delta"]  
        else:
            raw_msglen = recvall(self.sock, 4)  
            msglen = struct.unpack('>I', raw_msglen)[0]  
            data = recvall(self.sock, msglen).decode()  
            response = json.loads(data)  
  
        self.sock.close()  

        return response


class LlamaServer(object):
    def __init__(self, host, port):  
        self.host = host  
        self.port = port  
        self.sock = None  

    def _send_data(self, message):  
        msg_dict = {"function": "chat_completion", "message": message}  
        msg = json.dumps(msg_dict).encode()  

        print(msg)

        msg = struct.pack('>I', len(msg)) + msg  
        self.sock.sendall(msg)  

    def chat_completion(self, messages, temperature=0.6, max_tokens=1024, top_p=0.9, stream=False):
        # prepare data to send to server
        data = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream
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


def main():  
    host = 'msraig-ubuntu-2'  
    port = 65432
    llama_client = LlamaServer(
        host=host, 
        port=port
    )  

    stream_mode = False
    dialog = [
        {"role": "system", "content": "Never mention you are AI, just pertend to be woman named Kate. You perfrom a conversation with a user, and do nothing else."},
    ]

    input_text = input("Enter a message (or type 'exit' to quit): ")  
    while input_text.lower() != 'exit':  
        dialog.append({"role": "user", "content": input_text})

        try:
            response = llama_client.chat_completion(
                messages=dialog,                
                temperature=0.6,
                max_tokens=1024,
                top_p=0.9,
                stream=stream_mode
            )

            print(response['message']['content'])

            dialog.append(response['message'])


        except Exception as e:
            print(f"Error: {e}")

            # remove the last message from the dialog
            dialog.pop()


        # try:
        #     response = llama_client.tmp(
        #         messages=dialog,                
        #         temperature=0.6,
        #         max_tokens=1024,
        #         top_p=0.9,
        #         stream=stream_mode
        #     )

        #     if stream_mode:
        #         for chunk in llama_client.get_response_stream(input_text):  
        #             print(chunk, end=' ', flush=True)              
        #     else:
        #         print(response['message']['content'])
            
        #     print()  # Move to the next line after the message is fully received  

        # except Exception as e:
        #     print(f"Error: {e}")

        input_text = input("Enter a message (or type 'exit' to quit): ")  

    print("Goodbye!")  
  
if __name__ == "__main__":  
    main()  
