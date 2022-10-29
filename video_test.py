import codecs
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from io import BytesIO
from typing import List

import paddle
import argparse
import cv2
import numpy as np
import os
from models.model import FaceSwap, l2_norm
from models.arcface import IRBlock, ResNet
from utils.align_face import back_matrix, dealign, align_img, align_img_mask
from utils.util import paddle2cv, cv2paddle
from utils.prepare_data import LandmarkModel
from tqdm import tqdm

app = FastAPI()

def get_id_emb(id_net, id_img):
    id_img = cv2.resize(id_img, (112, 112))
    id_img = cv2paddle(id_img)
    mean = paddle.to_tensor([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1))
    std = paddle.to_tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1))
    id_img = (id_img - mean) / std

    id_emb, id_feature = id_net(id_img)
    id_emb = l2_norm(id_emb)

    return id_emb, id_feature

def load_img_into_np_array(data):
    return np.array(Image.open(BytesIO(data)))

@app.post('/api/uploadfiles/') 
async def video_test(original_video:  List[UploadFile] = File(description="Original Video"),
                    key_frame: UploadFile = File(description="Key Frame Image"),
                    reference_img: UploadFile = File(description="Reference Image"),
                    mask_img: UploadFile = File(description="Mask Image")):
    
    reference_img = load_img_into_np_array(await reference_img.read())
    swaped_reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
    #ref img 저장하기
    source_img_path = 'data/1.jpg'
    cv2.imwrite(source_img_path, swaped_reference_img)
    
    mask_img = load_img_into_np_array(await mask_img.read())
    swaped_mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("./mask/original_mask.png", swaped_mask_img)
    
    key_img = load_img_into_np_array(await key_frame.read())
    swaped_key_img = cv2.cvtColor(key_img, cv2.COLOR_BGR2RGB)
    
    #video_path 저장하기
    target_video_path = 'data/video.mp4'
    for f in original_video :
        contents = await f.read()
        with open(target_video_path, "wb") as fp :
            fp.write(contents)
    
    paddle.set_device("gpu")
    faceswap_model = FaceSwap(True)

    id_net = ResNet(block=IRBlock, layers=[3, 4, 23, 3])
    id_net.set_dict(paddle.load('./checkpoints/arcface.pdparams'))

    id_net.eval()

    weight = paddle.load('./checkpoints/MobileFaceSwap_224.pdparams')

    landmarkModel = LandmarkModel(name='landmarks')
    landmarkModel.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))
    id_img = cv2.imread(source_img_path)

    landmark = landmarkModel.get(id_img)
    if landmark is None:
        print('**** No Face Detect Error ****')
        exit()
    aligned_id_img, _ = align_img(id_img, landmark)

    id_emb, id_feature = get_id_emb(id_net, aligned_id_img)

    faceswap_model.set_model_param(id_emb, id_feature, model_weight=weight)
    faceswap_model.eval()
    
    #여기서 key frame align_img하면서 마스크도 같이 내기!(새로운 함수 구현해서 한꺼번에 나오게 하기!!)
    landmark = landmarkModel.get(swaped_key_img)
    att_img, tune_mask, _ = align_img_mask(swaped_key_img, swaped_mask_img, landmark) #원본 마스크랑 이미지 넣어서 사이즈 바꾸기
    
    #확인용
    cv2.imwrite("./mask/tune_mask.png", tune_mask)
    
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    cap = cv2.VideoCapture()
    cap.open(target_video_path)
    videoWriter = cv2.VideoWriter("results/result_video.mp4", fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    all_f = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    for i in tqdm(range(int(all_f))):
        ret, frame = cap.read()
        landmark = landmarkModel.get(frame)
        if landmark is not None:
            att_img, back_matrix = align_img(frame, landmark)
            att_img = cv2paddle(att_img)
            res, mask = faceswap_model(att_img)
            
            #type 맞추기
            tune_mask = np.reshape(tune_mask, (1,1,224,224)).astype(np.float32)
            tune_mask = tune_mask.tolist()
            mask = paddle.to_tensor(tune_mask) 
            
            res = paddle2cv(res)
            mask = np.transpose(mask[0].numpy(), (1, 2, 0))
            res = dealign(res, frame, back_matrix, mask)
            frame = res
        else:
            print('**** No Face Detect Error ****')
        videoWriter.write(frame)
    cap.release()
    videoWriter.release()
    
    def iterfile():  # 
        output_video_path = './results/result_video.mp4'
        output_video_path2 = './results/result_video2.mp4'
        os.system(f"ffmpeg -i  {output_video_path} -vcodec libx264 -y {output_video_path2}")
        with open(output_video_path2, mode="rb") as file_like:  # 
            yield from file_like  #     
    
    # return video object
    return StreamingResponse(iterfile(), media_type='video/mp4')

@app.get('/api/get-result/') 
async def give_move():
    def iterfile():  # 
        output_video_path = './results/result_video2.mp4'
        with open(output_video_path, mode="rb") as file_like:  # 
            
            yield from file_like  #     
    
    # return video object
    return StreamingResponse(iterfile(), media_type='video/mp4')