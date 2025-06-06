.PHONY: build

build:
	docker build -t sagemaker_shim .
	docker run \
		-v $(shell readlink -f ./dist/):/opt/src/dist/ \
		--rm \
		sagemaker_shim \
		bash -c "uv run make -C dist clean && uv run make -C dist release"
