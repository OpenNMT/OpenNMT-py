include .env
build:
	docker build --tag opennmt-py-dev:2.1.0-cuda12.1-cudnn8-devel .
up:
	docker run -it --name opennmt-py-dev --gpus=all --shm-size=10G \
			-v ${MODEL_FOLDER}:/workspace/model \
			opennmt-py-dev:2.1.0-cuda12.1-cudnn8-devel /bin/bash
down:
	docker stop opennmt-py-dev && docker rm opennmt-py-dev