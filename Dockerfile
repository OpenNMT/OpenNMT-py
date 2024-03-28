FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

RUN apt-get update
RUN apt-get install -y vim
COPY ./ /dependencies/opennmt-py/
RUN cd /dependencies/opennmt-py/ && pip install -e .
RUN pip install black
RUN pip install sentencepiece
