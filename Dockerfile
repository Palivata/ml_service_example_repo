FROM python:3.10

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt --no-cache-dir
WORKDIR /project
COPY src src
COPY configs configs
COPY app.py app.py
COPY models models
CMD ["python3", "app.py"]
