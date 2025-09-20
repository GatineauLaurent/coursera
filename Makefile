FREEIMGDIR ?= /opt/local
CUDADIR    ?= /usr/local/cuda
USE_NPP    ?= yes

 PROG_NAME := grayScaleConverterNPP
       CXX := g++
  CXXFLAGS := -std=c++17 -I$(FREEIMGDIR)/include
   LDFLAGS := -L$(FREEIMGDIR)/lib -lfreeimage

ifeq ($(USE_NPP), yes)
       NVCC := $(CUDADIR)/bin/nvcc
  NVCCFLAGS := -ccbin $(CXX) -m64 -gencode arch=compute_35,code=compute_35
   CXXFLAGS += -D_HAVE_NPP -I$(CUDADIR)/include
    LDFLAGS += -lnppc -lnppidei -lnppisu
endif

export USE_NPP PROG_NAME CXX CXXFLAGS LDFLAGS NVCC NVCCFLAGS CXXFLAGS LDFLAGS

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
