all: ../bin/$(PROG_NAME)

../bin/$(PROG_NAME): main.o process.o
	$(CXX) $(CXXFLAGS) -o $@ $^

main.o: main.cpp process.h
process.o: process.cpp process.h

clean:
	rm -f *.o
