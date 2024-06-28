#!/bin/bash
#SBATCH -A trn025
#SBATCH -N2
#SBATCH -J vector_addition_mpi_omp_GPU
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


srun -N2 $APPTAINER exec $CONTAINER amdclang++ -lmpi -fopenmp  --offload-arch=gfx90a -L${MPICH_DIR}/lib -I${MPICH_DIR}/include   ./vAdd_mpi_ompGPU.cpp -o vAdd_mpi_ompGPU 
srun -N2 $APPTAINER exec $CONTAINER ./vAdd_mpi_ompGPU

