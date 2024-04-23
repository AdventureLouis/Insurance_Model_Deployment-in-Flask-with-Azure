FROM python:3.9.13

WORKDIR /app

COPY './requirements.txt' .


RUN apt update -y

RUN apt-get update && pip install -r requirements.txt

COPY . . 


CMD ["python3","app.py"]
