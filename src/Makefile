all: ../bin/$(PROG_NAME)

../bin/$(PROG_NAME): main.o process.o
ifeq ($(USE_NPP), yes)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $^
else
	$(CXX) $(LDFLAGS) -o $@ $^
endif

main.o: main.cc process.h
process.o: process.cc process.h

clean:
	rm -f *.o
