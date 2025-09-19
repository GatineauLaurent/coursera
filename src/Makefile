all: ../bin/$(PROG_NAME)

../bin/$(PROG_NAME): main.o process.o
	$(CXX) $(CXXFLAGS) -o $@ $^

main.o: main.cc process.h
process.o: process.cc process.h

clean:
	rm -f *.o
