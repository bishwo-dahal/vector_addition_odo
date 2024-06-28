COMP  = CC
FLAGS = -fopenmp

vAdd_mpi_ompGPU: vAdd_mpi_ompGPU.o
	$(COMP) $(FLAGS) vAdd_mpi_ompGPU.o -o vAdd_mpi_ompGPU

vAdd_mpi_ompGPU.o: vAdd_mpi_ompGPU.cpp
	$(COMP) $(FLAGS) -c vAdd_mpi_ompGPU.cpp

.PHONY: clean

clean:
	rm -f vAdd_mpi_ompGPU *.o
