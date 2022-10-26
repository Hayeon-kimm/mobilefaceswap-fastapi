## [LAIT2 server]
1. 도커 컨테이너 실행하기<br/>
~~~
nvidia-docker run --name [이름] -p 8000:8000/tcp -it -v $PWD:/paddle paddlepaddle/paddle:2.3.2-gpu-cuda11.2-cudnn8  /bin/bash
~~~
2. 컨테이너에서 추가 모듈 설치<br/>
~~~
pip install fastapi opencv-python insightface==0.2.1 onnxruntime pillow
~~~
3. git-hub repository "model-serve-frontend"에서 프론트 서버 실행하기(https://github.com/SGM-StyleTransfer/model-serve-frontend
4. `uvicorn --host=0.0.0.0 --port 8000 video_test:app`으로 실행하기
<br/><br/>
## [docs로 test 버전]
1,2 동일, 3 생략 후 4로 접속 <br/>
이후 URL 마지막에 /docs를 입력하여 비디오, 이미지 입력 -> results에서 "result_video.mp4"로 실행결과 확인 가능
