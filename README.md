# VectorSequence

## Install

faiss:

git clone https://github.com/facebookresearch/faiss.git
cd faiss
cmake -DBUILD_TESTING=OFF -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DCMAKE_INSTALL_PREFIX=$HOME/local/faiss -B build .
make -C build -j faiss
make -C build install

## Build & Run

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
./vss_test 768 ../../datasets/droid/vectors-dinov2/64-32-Uni_8_16-10-100 hnsw



for windows:

cmake -G "MinGW Makefiles" 