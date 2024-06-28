CCOMP  = CC
CLFAGS = -fopenmp

GCOMP  = hipcc
GFLAGS = --amdgpu-target=gfx90a

LFLAGS = -fopenmp

INCLUDES  = 
LIBRARIES = -L${ROCM_PATH}/lib -lamdhip64

vAdd_hip_ompCPU: vAdd_ompCPU.o vAdd_hip.o
	${CCOMP} ${LFLAGS} ${LIBRARIES} vAdd_ompCPU.o vAdd_hip.o -o vAdd_hip_ompCPU

vAdd_hip.o: vAdd_hip.cpp vAdd_hip.h
	${GCOMP} ${GFLAGS} ${INCLUDES} -c vAdd_hip.cpp

vAdd_ompCPU.o: vAdd_ompCPU.cpp
	${CCOMP} ${CLFAGS} ${INCLUDES} -c vAdd_ompCPU.cpp

.PHONY: clean

clean:
	rm -f vAdd_hip_ompCPU *.o
