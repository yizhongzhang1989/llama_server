import requests

def send_request_to_server(input_text):
    # The URL of the server's endpoint
    url = "http://msraig-ubuntu-2:5000/infer"  # Replace with your actual server address and port
    # Send a POST request with JSON data
    response = requests.post(url, json={'text': input_text})
    # Get the response data
    if response.status_code == 200:
        return response.json()['response']
    else:
        return None

# Example usage
response = send_request_to_server("Hello, world!")
print(response)
