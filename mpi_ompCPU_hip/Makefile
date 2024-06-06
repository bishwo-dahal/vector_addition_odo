MPI_COMP  = CC
FLAGS     = -fopenmp
LIBRARIES = -L${ROCM_PATH}/lib -lamdhip64
INCLUDES  = 

HIP_COMP  = hipcc
HIP_FLAGS = --amdgpu-target=gfx90a

vector_addition: vAdd.o vAdd_gpu.o
	$(MPI_COMP) $(FLAGS) $(LIBRARIES) vAdd.o vAdd_gpu.o -o vector_addition

vAdd.o: vAdd.cpp
	$(MPI_COMP) $(FLAGS) $(INCLUDES) -c vAdd.cpp

vAdd_gpu.o: vAdd_gpu.cpp
	$(HIP_COMP) $(HIP_FLAGS) $(INCLUDES) -c vAdd_gpu.cpp

.PHONY: clean

clean:
	rm -f vector_addition *.o

