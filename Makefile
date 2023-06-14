.PHONY: build

build:
	docker build -t sagemaker_shim .
	docker run \
		-v $(shell readlink -f ./sagemaker_shim/):/opt/src/sagemaker_shim/ \
		-v $(shell readlink -f ./dist/):/opt/src/dist/ \
		-v $(shell readlink -f ./tests/):/opt/src/tests/ \
		--rm \
		sagemaker_shim \
		bash -c "poetry run make -C dist clean && poetry run make -C dist release"
