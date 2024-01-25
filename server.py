from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

@app.route('/')  
def hello_world():  
    return 'Hello, World!'  

@app.route('/infer', methods=['POST'])
def infer():
    # Get the request data
    data = request.json
    input_text = data['text']
    
    # Return the response
    return jsonify({'response': input_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
