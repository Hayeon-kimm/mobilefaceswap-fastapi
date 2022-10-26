from fastapi import FastAPI, File, UploadFile
import shutil, pickle
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU:5')
print(gpus)
# for i in range(len(gpus)): #모든 gpu에 동일한 양 할당하기
# 	tf.config.experimental.set_memory_growth(gpus[i], True)

app = FastAPI()


@app.post('/') #데이터 업로드 및 현재 실행 경로에 저장
async def root(file: UploadFile = File(...)) :
    with open(f'{file.filename}', "wb") as buffer :
        shutil.copyfileobj(file.file, buffer)
    print(type(file))
    return{"file_name" : file.filename}

#모델 로드
pickle_in = open("model_220915.pkl","rb")
model = pickle.load(pickle_in)

#여기서 문제는 file이 <class 'starlette.datastructures.UploadFile'>인데 우리는 numpy로 바꾸어줘야함!
def load_image_into_numpy_array(data): #!!
    return np.array(Image.open(BytesIO(data)))

@app.post('/predict') #파이프라인, image를 함수안에넣는법!!
#가능한 올리자마자 바로 넣는(굳이 저장안하고.. -> 매개변수로 연결)
async def predict_num(img: UploadFile = File(...)):
    img = load_image_into_numpy_array(await img.read())
    img = np.resize(img,(28,28))
    img = np.expand_dims(img,0) #1차원의 모델은 keras에 의해 predict가 불가능해서 2차원으로 만들어주는 작업
    predictions=model.predict(img)
    best = np.argmax(predictions)
    return { "Answer" : int(best) } #fastapi는 numpy를 return 못한다!

@app.post('/add')
async def add(a,b) :
    return {"Answer" : int(a)+int(b)}

