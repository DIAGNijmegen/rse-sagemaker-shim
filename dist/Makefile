TEST_VERSION=$(shell python -c "from tests import __version__;print(__version__)")
RELEASE_VERSION="$(shell python -c "from importlib.metadata import version;print(version('sagemaker-shim'))")-$(shell uname -s)-$(shell uname -i)"
DIST_DIR=./dist

.PHONY: clean all release

all: crane sagemaker-shim-static-${TEST_VERSION}

sagemaker-shim-${TEST_VERSION}:
	pyinstaller -F ../sagemaker_shim/cli.py --paths .. --distpath . -n "sagemaker-shim-${TEST_VERSION}"

sagemaker-shim-static-${TEST_VERSION}: sagemaker-shim-${TEST_VERSION}
	staticx "sagemaker-shim-${TEST_VERSION}" "sagemaker-shim-static-${TEST_VERSION}"

crane-src.tar.gz:
	wget https://github.com/google/go-containerregistry/releases/download/v0.9.0/go-containerregistry_Linux_x86_64.tar.gz -O crane-src.tar.gz && \
	echo "1d2cf3fac0830dd8e5fb6e2829fdfc304a3d4a0f48f7e1df9ebb7e2921f28803  crane-src.tar.gz" | shasum -c - || exit 1

crane: crane-src.tar.gz
	tar -xzvf crane-src.tar.gz crane && \
	chmod u+x crane

clean:
	rm -rf build dist crane crane-src.tar.gz sagemaker-shim* checksums.txt

release: sagemaker-shim-static-${TEST_VERSION}
	cp "sagemaker-shim-static-${TEST_VERSION}" "sagemaker-shim-${RELEASE_VERSION}"
	tar -zcvf "sagemaker-shim-${RELEASE_VERSION}.tar.gz" "sagemaker-shim-${RELEASE_VERSION}"
	sha256sum "sagemaker-shim-${RELEASE_VERSION}.tar.gz" > checksums.txt
