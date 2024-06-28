COMP  = g++
FLAGS = -fopenmp -lamdhip64  -L/opt/rocm-5.5.1/lib -I/opt/rocm-5.5.1/include/ -D__HIP_PLATFORM_AMD__

vAdd_hip_ompCPU: vAdd_hip_ompCPU.o
	ls -lrt && $(COMP) $(FLAGS)  vAdd_hip_ompCPU.o -o vAdd_hip_ompCPU

vAdd_hip_ompCPU.o: vAdd_hip_ompCPU.cpp
	$(COMP)  $(FLAGS)  vAdd_hip_ompCPU.cpp -o vAdd_hip_ompCPU.o 

.PHONY: clean

clean:
	rm -f vAdd_hip_ompCPU *.o
