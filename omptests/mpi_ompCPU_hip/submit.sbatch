#!/bin/bash
#SBATCH -A trn025
#SBATCH -N2
#SBATCH -J vector_addition_mpi_omp_hip
#SBATCH -o res/%x_%j.out
#SBATCH -e res/%x_%j.err
#SBATCH -t 3:00



module load cce
module load amd-mixed/5.7.1
module load cray-mpich/8.1.23

export MPICH_GPU_SUPPORT_ENABLED=1

set -ex

export CONTAINER=../containers/opensusempich342rocm571.sif
export APPTAINER=apptainer
export MPICH_DIR=/opt/mpich


srun -N2 $APPTAINER exec $CONTAINER hipcc -fopenmp -I${MPICH_DIR}/include -L${MPICH_DIR}/lib   -lmpi ./vAdd.cpp ./vAdd_gpu.cpp -o vAdd_mpi_ompCPU_hip

srun -N2 $APPTAINER exec $CONTAINER ./vAdd_mpi_ompCPU_hip

