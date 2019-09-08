FROM jupyter/pyspark-notebook
USER root

WORKDIR /juypter
COPY . ./

RUN chown -R jovyan /juypter && \
    chmod -R 777 ./
RUN pip install --upgrade pip && \
  pip3 install -r requirements.txt

