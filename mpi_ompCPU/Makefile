COMP  = CC
FLAGS = -fopenmp

vAdd_mpi_ompCPU: vAdd_mpi_ompCPU.o
	$(COMP) $(FLAGS) vAdd_mpi_ompCPU.o -o vAdd_mpi_ompCPU

vAdd_mpi_ompCPU.o: vAdd_mpi_ompCPU.cpp
	$(COMP) $(FLAGS) -c vAdd_mpi_ompCPU.cpp

.PHONY: clean

clean:
	rm -f vAdd_mpi_ompCPU *.o
