FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime

WORKDIR /home/$USER

COPY requirements.txt ./

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install build-essential -y
RUN pip install --no-cache-dir --upgrade pip \
	&& pip install --no-cache-dir -r requirements.txt