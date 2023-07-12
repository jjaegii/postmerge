# tensorflow 이미지를 베이스로 저번 과제 파일 합쳐서 이미지 만들기
FROM ubuntu:20.04
LABEL maintainer "jjaegii <hn06038@gmail.com>"
RUN apt-get update && apt-get install python3 python3-pip -y
WORKDIR /src
COPY train.py .
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# 이미지 빌드 후
# docker run -it -v ./100_1:/src/100_1 -v ./100_2:/src/100_2 -v ./model:/src/model 이미지이름:태그
ENTRYPOINT [ "python3" ]
CMD [ "train.py" ]