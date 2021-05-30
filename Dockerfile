FROM python:3.6-slim-stretch
RUN mkdir /job/
COPY requirements.txt /job/requirements.txt
RUN pip install --no-cache-dir -r /job/requirements.txt
COPY train.py /job/train.py

WORKDIR /job

ENTRYPOINT ["python", "train.py"]
