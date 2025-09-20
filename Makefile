export   USE_NPP ?= yes
export PROG_NAME := grayScaleConverterNPP
export       CXX := g++
export  CXXFLAGS := -std=c++17 -I/opt/local/include
export   LDFLAGS := -L/opt/local/lib -lfreeimage

ifeq ($(USE_NPP), yes)
  export      NVCC := /usr/local/cuda/bin/nvcc
  export NVCCFLAGS := -ccbin $(CXX) -m64 -gencode arch=compute_35,code=compute_35
  export  CXXFLAGS += -D_HAVE_NPP
  export   LDFLAGS += -lnppc
endif

all: bin/$(PROG_NAME)

bin/$(PROG_NAME): src/*.cc src/*.h
	$(MAKE) -C src

run: bin/$(PROG_NAME)
	bin/$(PROG_NAME) --input data/* --output_dir output

clean:
	rm -f bin/$(PROG_NAME)
	$(MAKE) -C src clean

help:
	@echo "Available make commands:"
	@echo "  make        - Build the project."
	@echo "  make run    - Run the project."
	@echo "  make clean  - Clean up the build files."
	@echo "  make help   - Display this help message."
