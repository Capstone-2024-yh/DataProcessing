from fastapi import FastAPI
from word2vec import get_word_vector, load_model
from contextlib import asynccontextmanager
from pydantic import BaseModel
import numpy as np


# lifespan 이벤트 핸들러를 사용하여 애플리케이션 시작 시 작업 실행
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model() # 모델 로드
    print("모델 로드 완료")

    yield
    # 필요 시 애플리케이션 종료 시 정리 작업을 수행
    print("애플리케이션 종료")

app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# 이거는 그냥 확인용 실제로는 사용 안함
@app.get("/word2vec/{word}")
async def get_vector(word: str):
    raw_data = list(get_word_vector(word))
    # print(raw_data)
    data = list(map(float, raw_data))
    # print(data)
    return {
        "word": word,
        "vector": data
    }


class Word2VecRequest(BaseModel):
    words: list[str]

@app.post("/word2vec") 
async def get_vectors(request: Word2VecRequest):
    words = request.words
    result = {}
    for word in words:
        raw_data: np.ndarray
        if '/' in word:
            split_data = word.split('/')
            if split_data[0] == "X":
                # X로 시작하면 반대벡터 연산 넣은거
                raw_data = - get_word_vector(split_data[1])
            else :
                raw_data = get_word_vector(split_data[1])
        else:
            # 그냥 하는거
            raw_data = get_word_vector(word)

        # 노멀라이제이션을 위한 l2 norm 구하기
        l2_norm = np.linalg.norm(raw_data)
        normalized_data = raw_data / l2_norm

        result[word] = normalized_data.tolist()
    return result