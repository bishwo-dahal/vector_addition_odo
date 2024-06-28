COMP    = CC
FLAGS   = -fopenmp

vAdd_ompGPU: vAdd_ompGPU.o
	${COMP} ${FLAGS} vAdd_ompGPU.o -o vAdd_ompGPU

vAdd_ompGPU.o: vAdd_ompGPU.cpp
	${COMP} ${FLAGS} -c vAdd_ompGPU.cpp

.PHONY: clean

clean:
	rm -f vAdd_ompGPU *.o
