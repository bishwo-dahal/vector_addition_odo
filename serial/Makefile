COMP  = CC
FLAGS =

vAdd_serial: vAdd_serial.o
	$(COMP) $(FLAGS) vAdd_serial.o -o vAdd_serial

vAdd_serial.o: vAdd_serial.cpp
	$(COMP) $(FLAGS) -c vAdd_serial.cpp

.PHONY: clean

clean:
	rm -f vAdd_serial *.o
