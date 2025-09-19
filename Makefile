export PROG_NAME := grayScaleConverterNPP
export CXX := g++
export CXXFLAGS := -std=c++17

all: bin/$(PROG_NAME)

bin/$(PROG_NAME): src/*.cpp src/*.h
	$(MAKE) -C src

run: bin/$(PROG_NAME)
	bin/$(PROG_NAME) --input data/* --output_dir output

clean:
	rm -f bin/$(PROG_NAME)
	$(MAKE) -C src clean

install: bin/$(PROG_NAME)
	@echo "No installation required."
