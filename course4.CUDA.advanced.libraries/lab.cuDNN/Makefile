# IDIR=./
CXX = g++

all: clean build

build: cudnn_example.cpp
	$(CXX) cudnn_example.cpp -o cudnn_example.exe -std=c++17 \
	-I/usr/local/cuda/include \
	-I/usr/local/cuda/targets/x86_64-linux/include \
	-L/usr/local/cuda/lib64 \
	-lcudart -lcuda -lcudnn

run:
	./cudnn_example.exe $(ARGS)

clean:
	rm -f cudnn_example.exe output*.txt
