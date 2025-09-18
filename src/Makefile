all: ../bin/$(PROG_NAME)

../bin/$(PROG_NAME): main.c
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f *.o
