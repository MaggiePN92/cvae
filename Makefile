# Config
IMAGE := pytorch/pytorch
CONTAINER_NAME := in5310
PROJECT_PATH := /project2
RUN := docker exec -it $(CONTAINER_NAME)

.PHONY: all pull docker install run clean

all: pull docker install run

pull:
	docker pull $(IMAGE)

docker:
	- docker start $(CONTAINER_NAME) >/dev/null 2>&1 || \
	docker run -d -i \
		-v $(PWD):$(PROJECT_PATH) \
		-w $(PROJECT_PATH) \
		--name $(CONTAINER_NAME) --init \
		$(IMAGE)

install:
	$(RUN) pip install --user -r docker-requirements.txt

run:
	$(RUN) python validate_project2.py

clean:
	- docker stop $(CONTAINER_NAME)
	- docker rm $(CONTAINER_NAME)
