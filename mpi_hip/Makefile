COMP   = CC
CFLAGS = -std=c++11 -D__HIP_ROCclr__ -D__HIP_ARCH_GFX90A__=1 --rocm-path=${ROCM_PATH} --offload-arch=gfx90a -x hip
LFLAGS = -std=c++11 -D__HIP_ROCclr__ --rocm-path=${ROCM_PATH}

INCLUDES  = -I${ROCM_PATH}/include
LIBRARIES = -L${ROCM_PATH}/lib -lamdhip64 -lhsa-runtime64

vAdd_mpi_hip: vAdd_mpi_hip.o
	$(COMP) $(LFLAGS) ${LIBRARIES} vAdd_mpi_hip.o -o vAdd_mpi_hip

vAdd_mpi_hip.o: vAdd_mpi_hip.cpp
	$(COMP) $(CFLAGS) ${INCLUDES} -c vAdd_mpi_hip.cpp

.PHONY: clean

clean:
	rm -f vAdd_mpi_hip *.o
