#!/bin/bash
#SBATCH -A trn025
#SBATCH -N2
#SBATCH -J Vector_addition_on_cpu_OMP_HIP_separate_files
#SBATCH -t 15:00
#SBATCH -o /res/%x_%j.out
#SBATCH -e /res/%x_%j.err


module load cce
module load amd-mixed/5.7.1
module load cray-mpich/8.1.23

export MPICH_GPU_SUPPORT_ENABLED=1

export CONTAINER=../containers/opensusempich342rocm571.sif
export APPTAINER=apptainer

srun -N2 $APPTAINER exec --rocm  $CONTAINER hipcc -c ./vAdd_hip.cpp ./vAdd_ompCPU.cpp
srun -N2 $APPTAINER exec --rocm  $CONTAINER hipcc  -fopenmp  vAdd_ompCPU.o vAdd_hip.o -o vAdd_hip_ompCPU
srun -N2 apptainer exec $CONTAINER ./vAdd_hip_ompCPU


