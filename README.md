# Dockerfile 실행방법

docker build -t 이미지이름:태그 .

빌드 후

docker run -it -v ./100_1:/src/100_1 -v ./100_2:/src/100_2 -v ./model:/src/model 이미지이름:태그