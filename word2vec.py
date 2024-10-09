import fasttext
import fasttext.util

# 사전 학습된 한국어 FastText 모델 로드
model_path = 'cc.ko.300.bin'

def load_model():
    global model
    model = fasttext.load_model(model_path)

def get_word_vector(word):
    return model.get_word_vector(word)