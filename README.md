## [LAIT2 server]
*  주로 사용할 파일은 video_test.py 입니다.
* 이미지명:태그는 제가 paddle 이란 이름으로 이미지를 주었기 때문에 paddle일 거에요! 정확하 이름명으 알고 싶으며 이미지르 다 설치하 후 docker images 명령어로 이미지 및 컨테이너 이름을 확인하세요!
1. 도커 컨테이너 실행하기(PWD이므로 미리 mobilefaceswap-fastapi 디렉토리에서 이 명령어를 실행하세요! 그래야 이 폴더와 volume 연결이 됩니다)<br/>
~~~
nvidia-docker run --name [컨테이너 지정할 이름] -p 8000:8000/tcp -it -v $PWD:/paddle [이미지명]:[태그]  /bin/bash
~~~
2. 컨테이너에서 추가 모듈 설치(이미지 자체를 전달했기 때문에 모두 있을거에요 -> fastapi --version으로 확인해보세용! 존재한다면 이 과정은 생략하세요)<br/>
~~~
pip install fastapi opencv-python insightface==0.2.1 onnxruntime pillow uvicorn
~~~
3. 컨테이너에서 codec 변환 패키지르 설치하기 + results(결과저장), data(input저장) 폴더 생성
~~~
apt-get update
apt-get install ffmpeg x264 libx264-dev
mkdir data results
~~~
4. git-hub repository "model-serve-frontend"에서 프론트 서버 실행하기(https://github.com/Hayeon-kimm/model-serve-frontend)
5. 기존 도커 컨테이너에서 paddle 폴더로 들어가서 `uvicorn --host=0.0.0.0 --port 8000 video_test:app`으로 실행하기
~~~
cd ..
cd paddle
uvicorn --host=0.0.0.0 --port 8000 video_test:app
~~~

<br/><br/>
## [docs로 test 버전] -> 안하셔도 됩니다 !
1,2 동일, 3 생략 후 4로 접속 <br/>
이후 URL 마지막에 /docs를 입력하여 비디오, 이미지 입력 -> results에서 "result_video.mp4"로 실행결과 확인 가능
