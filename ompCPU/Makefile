COMP  = CC
FLAGS = -fopenmp

vAdd_ompCPU: vAdd_ompCPU.o
	$(COMP) $(FLAGS) vAdd_ompCPU.o -o vAdd_ompCPU

vAdd_ompCPU.o: vAdd_ompCPU.cpp
	$(COMP) $(FLAGS) -c vAdd_ompCPU.cpp

.PHONY: clean

clean:
	rm -f vAdd_ompCPU *.o
