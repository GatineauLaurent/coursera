export PROG_NAME := grayScaleConverterNPP
export CXX := g++
export CXXFLAGS := -std=c++17 -I/opt/local/include
export LDFLAGS := -L/opt/local/lib -lfreeimage
export NVCC := /usr/local/cuda/bin/nvcc
export NVCCFLAGS := -ccbin $(CXX) -m64 -gencode arch=compute_35,code=compute_35

all: bin/$(PROG_NAME)

bin/$(PROG_NAME): src/*.cc src/*.h
	$(MAKE) -C src

run: bin/$(PROG_NAME)
	bin/$(PROG_NAME) --input data/* --output_dir output

clean:
	rm -f bin/$(PROG_NAME)
	$(MAKE) -C src clean

install: bin/$(PROG_NAME)
	@echo "No installation required."
