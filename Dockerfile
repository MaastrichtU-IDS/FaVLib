FROM jupyter/pyspark-notebook:814ef10d64fb
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
  pip install -r requirements.txt


