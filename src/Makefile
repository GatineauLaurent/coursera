all: ../bin/$(PROG_NAME)

../bin/$(PROG_NAME): main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f *.o
