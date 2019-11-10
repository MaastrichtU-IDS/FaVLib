FROM jupyter/pyspark-notebook
USER root

WORKDIR /jupyter
COPY requirements.txt ./

RUN chown -R jovyan /jupyter && \
    chmod -R 777 ./

RUN apt-get update && apt-get install -y \
    wget \
    ca-certificates \
    build-essential \
    curl \ 
    cwltool

RUN pip install --upgrade pip && \
  pip3 install -r requirements.txt


