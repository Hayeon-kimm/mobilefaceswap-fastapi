from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from io import BytesIO
from typing import List

import torch
import logging
import paddle
import argparse
import cv2
import numpy as np
import os
import glob
from models.model import FaceSwap, l2_norm
from models.arcface import IRBlock, ResNet
from utils.align_face import back_matrix, dealign, align_img
from utils.util import paddle2cv, cv2paddle
from utils.prepare_data import LandmarkModel
from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio
from skimage.color import rgb2ycbcr


from models.RSTT.models import create_model
from models.RSTT.utils import (mkdirs, parse_config, AverageMeter, structural_similarity, 
                   read_seq_images, index_generation, setup_logger, get_model_total_params)


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


###########요 밑에 함수 추가해주었습니다!################################################################################################

def set_for_RSTT():
    # parser = argparse.ArgumentParser(description='Space-Time Video Super-Resolution Evaluation on Vimeo90k dataset')
    # parser.add_argument('--config', type=str, help='Path to config file (.yaml).')
    # args = parser.parse_args()
    # config = parse_config(args.config, is_train=False)

    # save_path = "./" 
    # mkdirs(save_path)
    # setup_logger('base', save_path, 'test', level=logging.INFO, screen=True, tofile=True)
    model = create_model(config)
    model_params = get_model_total_params(model)

    logger = logging.getLogger('base')
    logger.info('use GPU {}'.format(config['gpu_ids']))
    logger.info('Data: {} - {} - {}'.format(config['dataset']['name'], config['dataset']['mode'], config['dataset']['dataset_root']))
    logger.info('Model path: {}'.format(config['path']['pretrain_model']))
    logger.info('Model parameters: {} M'.format(model_params))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(config['path']['pretrain_model']), strict=True)
    model.eval()
    model = model.to(device)

    
    config['dataset']['dataset_root'] = "./simple_test_dataset" 

    # LR_paths = sorted(glob.glob(os.path.join(config['dataset']['dataset_root'], config['dataset']['mode']+'_test', '*')))
    LR_paths = "inter_results/"
    
    return LR_paths, config, model, device

#####################################################################################################################################################

@app.post('/api/uploadfiles/') #해결!!
async def video_test(original_video:  List[UploadFile] = File(description="Original Video"),
                    #key_frame: UploadFile = File(description="Key Frame Image"),
                    reference_img: UploadFile = File(description="Reference Image"),):
                    #mask_img: UploadFile = File(description="Mask Image")):
    
    reference_img = load_img_into_np_array(await reference_img.read())
    swaped_reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
    #ref img 저장하기
    source_img_path = 'data/1.jpg'
    cv2.imwrite(source_img_path, swaped_reference_img)
    
    #video_path 저장하기
    target_video_path = 'data/video.mp4'
    for f in original_video :
        contents = await f.read()
        with open(target_video_path, "wb") as fp :
            fp.write(contents)
    #mask는 조금 있다가 해보기!
    
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

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture()
    cap.open(target_video_path)
    videoWriter = cv2.VideoWriter("results/result_video.mp4", fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    inter_results = "inter_results/"

    all_f = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    for i in tqdm(range(int(all_f))):
        ret, frame = cap.read()
        landmark = landmarkModel.get(frame)
        if landmark is not None:
            att_img, back_matrix = align_img(frame, landmark)
            att_img = cv2paddle(att_img)
            res, mask = faceswap_model(att_img)
            res = paddle2cv(res)
            mask = np.transpose(mask[0].numpy(), (1, 2, 0))
            res = dealign(res, frame, back_matrix, mask)
            frame = res

            #여기에서 이미지들을 다 저장을 함! for문 안에서는 video writing을 하지 않고!#############################################################################
            cv2.imwrite(inter_results + '/' + str(i).zfill(5) + '.jpeg', frame) #이걸 저장한 폴더를 (path, output)

        else:
            print('**** No Face Detect Error ****')
        # videoWriter.write(frame)
    
    LR_paths, config, rstt_model, device = set_for_RSTT()
    
    ##############여기부터####################################################################################################################################################
    for LR_path in LR_paths:

        tested_index = []
        imgs_LR = read_seq_images(LR_path)
        imgs_LR = imgs_LR.astype(np.float32) / 255.
        imgs_LR = torch.from_numpy(imgs_LR).permute(0, 3, 1, 2).contiguous()
        indices_list = index_generation(config['dataset']['num_out_frames'], imgs_LR.shape[0])
            
        for indices in indices_list:

            inputs = imgs_LR[indices[::2]].unsqueeze(0).to(device)
                
            with torch.no_grad():
                outputs = rstt_model(inputs)
            outputs = outputs.cpu().squeeze().clamp(0, 1).numpy()
                
            # PSNR, SSIM for each frame
            for idx, frame_idx in enumerate(indices):
                if frame_idx in tested_index:
                    continue
                tested_index.append(frame_idx)
                    
                output = (outputs[idx].squeeze().transpose((1, 2, 0)) * 255.0).round().astype(np.uint8)
                # output_y = rgb2ycbcr(output)[..., 0]

                cv2.imwrite(os.path.join(results + str(i).zfill(5) + '.jpeg'), output[...,::-1])


    #여기서 for문 안에서 안해줬던 video writing을 해줘야 함!!

    # 다시 이제 또 경로에서 하나하나 읽어서, 이제 video로 만들어줘야 함..

    vid_paths = 'results/'

    frames = [frame for frame in os.listdir(vid_paths) if frame.endswith(".jpg")]
    frame = cv2.imread(os.path.join(vid_path, frames[0]))
    height, width, layers = frame.shape

    videoWriter.write(cv2.imread(os.path.join(vid_paths, frame)))

    ################여기까지 수정했습니당!!####################################################################################################

    cap.release()
    videoWriter.release()
    def iterfile():  # 
        with open("results/result_video.mp4", mode="rb") as file_like:  # 
            yield from file_like  #     
    
    # return video object
    return StreamingResponse(iterfile(), media_type='video/mp4')


# if __name__ == '__main__':

#     parser = argparse.ArgumentParser(description="MobileFaceSwap Test")

#     parser = argparse.ArgumentParser(description="MobileFaceSwap Test")
#     parser.add_argument('--source_img_path', type=str, help='path to the source image')
#     parser.add_argument('--target_video_path', type=str, help='path to the target video')
#     parser.add_argument('--output_path', type=str, default='results', help='path to the output videos')
#     parser.add_argument('--image_size', type=int, default=224,help='size of the test images (224 SimSwap | 256 FaceShifter)')
#     parser.add_argument('--merge_result', type=bool, default=True, help='output with whole image')
#     parser.add_argument('--use_gpu', type=bool, default=False)

#     args = parser.parse_args()
#     video_test(args)
