TEST_VERSION=$(shell python -c "from tests import __version__;print(__version__)")
RELEASE_VERSION="$(shell uv version --short)-$(shell uname -s)-$(shell uname -m)"
DIST_DIR=./dist

.PHONY: clean all release

all: crane sagemaker-shim-static-${TEST_VERSION}

sagemaker-shim-${TEST_VERSION}:
	pyinstaller -F ../sagemaker_shim/cli.py --paths .. --distpath . -n "sagemaker-shim-${TEST_VERSION}"

sagemaker-shim-static-${TEST_VERSION}: sagemaker-shim-${TEST_VERSION}
	staticx "sagemaker-shim-${TEST_VERSION}" "sagemaker-shim-static-${TEST_VERSION}"

crane-src.tar.gz:
	curl -L https://github.com/google/go-containerregistry/releases/download/v0.19.2/go-containerregistry_Linux_x86_64.tar.gz -o crane-src.tar.gz && \
	echo "8e5a27413b3fa3b95467c8d36dfb55f7109075872a05fcc0dd674f537439513d  crane-src.tar.gz" | shasum -c - || exit 1

crane: crane-src.tar.gz
	tar -xzvf crane-src.tar.gz crane && \
	chmod u+x crane

clean:
	rm -rf build dist crane crane-src.tar.gz sagemaker-shim* checksums.txt

release: sagemaker-shim-static-${TEST_VERSION}
	cp "sagemaker-shim-static-${TEST_VERSION}" "sagemaker-shim-${RELEASE_VERSION}"
	tar -zcvf "sagemaker-shim-${RELEASE_VERSION}.tar.gz" "sagemaker-shim-${RELEASE_VERSION}"
	sha256sum "sagemaker-shim-${RELEASE_VERSION}.tar.gz" > checksums.txt
