export PROG_NAME := grayScaleConverterNPP

all: bin/$(PROG_NAME)

bin/$(PROG_NAME):
	$(MAKE) -C src

run: bin/$(PROG_NAME)
	bin/$(PROG_NAME)

clean:
	rm -f bin/$(PROG_NAME)
	$(MAKE) -C src clean

install: bin/$(PROG_NAME)
	@echo "No installation required."
