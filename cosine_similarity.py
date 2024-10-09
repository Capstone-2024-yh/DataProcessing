import numpy as np
from word2vec import model

# 두 벡터 간 코사인 유사도 계산 함수
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# word1 = "해수욕장"
# word2 = "바닷가"

word1 = "멀티탭"
word2 = "충전기"


# 두 단어에 대한 벡터
vector1 = model.get_word_vector(word1)
vector2 = model.get_word_vector(word2)

# 유사도 계산
similarity = cosine_similarity(vector1, vector2)
print(f"'{word1}'와 '{word2}'의 유사도: {similarity}")
