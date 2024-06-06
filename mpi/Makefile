COMP  = CC
FLAGS =

vAdd_mpi: vAdd_mpi.o
	${COMP} ${FLAGS} vAdd_mpi.o -o vAdd_mpi

vAdd_mpi.o: vAdd_mpi.cpp
	${COMP} ${FLAGS} -c vAdd_mpi.cpp

.PHONY: clean

clean:
	rm -f vAdd_mpi *.o
