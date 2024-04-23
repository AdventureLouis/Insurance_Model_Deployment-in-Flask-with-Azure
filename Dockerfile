FROM python:3.9.13

WORKDIR /app

COPY './requirements.txt' .

RUN pip install -r requirements.txt

RUN apt update -y

COPY . . 


CMD [ "python3","app.py" ]
