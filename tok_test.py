from llama.tokenizer import Tokenizer
import time


tokenizer = Tokenizer('tokenizer.model')
sp_model = tokenizer.sp_model

texts = [
    '句它     句它',
    '    句它     句它',
    "Hello, world! This is a test sentence in English. It includes various punctuation, such as commas, periods, and exclamation marks!",
    "Bonjour le monde! Voici une phrase test en français. Elle comprend diverses ponctuations, comme des virgules, des points et des points d'exclamation!",
    "你好，世界！这是一句中文的测试句子。它包括各种标点符号，如逗号、句号和感叹号！",
    "こんにちは、世界！これは日本語のテスト文です。コンマ、ピリオド、エクスクラメーションマークなど、さまざまな句読点が含まれています！",
    "안녕하세요, 세계! 이것은 한국어로 된 테스트 문장입니다. 쉼표, 마침표, 느낌표 등 다양한 구두점이 포함되어 있습니다!",
    "Hallo Welt! Dies ist ein Testsatz auf Deutsch. Es enthält verschiedene Satzzeichen, wie Kommas, Punkte und Ausrufezeichen!",
    "Ciao mondo! Questa è una frase di prova in italiano. Include varie punteggiature, come virgole, punti e punti esclamativi!",
    "Привет, мир! Это тестовое предложение на русском языке. Оно включает в себя различную пунктуацию, такую как запятые, точки и восклицательные знаки!",
    "Olá, mundo! Esta é uma frase de teste em português. Inclui várias pontuações, como vírgulas, pontos e pontos de exclamação!",
    "Hello, 世界！This is a mixed language test sentence. It includes various punctuation, such as commas, periods, and exclamation marks!",
]


for text in texts:
    start_time = time.time()
    tokens = sp_model.encode(text)
    decoded_text = sp_model.decode(tokens)
    end_time = time.time()
    print(f" {text}")
    print(f" {decoded_text}")
    
    # print('tokens: ', tokens)    

    def print_token_buffer(token_buffer):
        if sp_model.IdToPiece(token_buffer[0]).startswith('▁'): 
            print(' ', end='')

        tex_buf = sp_model.Decode(token_buffer)
        token_buffer = []
        print(tex_buf, end='')
        return token_buffer
    
    token_buffer = []

    for token in tokens:
        piece = sp_model.IdToPiece(token)

        if piece.startswith('▁'): 
            # the piece is a new word, print the buffer
            if len(token_buffer) > 0:
                token_buffer = print_token_buffer(token_buffer)
            # record this token
            token_buffer.append(token)
        else:
            # check whether the decoded text is ascii
            text = sp_model.Decode(token)

            if text.isascii():
                # the piece is a part of the previous word, append token to buffer
                token_buffer.append(token)
            elif text == piece:
                # the piece is not ascii, and the decoded text is a complete word, print the buffer
                if len(token_buffer) > 0:
                    token_buffer = print_token_buffer(token_buffer)
                # record this token
                token_buffer.append(token)
            else:
                # the piece is not ascii, and the decoded text is not a complete word, append token to buffer
                token_buffer.append(token)

    if len(token_buffer) > 0:
        token_buffer = print_token_buffer(token_buffer)
    print('')

