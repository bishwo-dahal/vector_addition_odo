#!/bin/bash
#SBATCH -A trn025
#SBATCH -N2
#SBATCH -J vector_addition_mpi
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

srun -N2 $APPTAINER exec --rocm  $CONTAINER amdclang++ -fopenmp -I/usr/lib64/gcc/x86_64-suse-linux/7/include  vAdd_ompGPU.cpp -o vAdd_ompGPU


srun -N2 $APPTAINER exec --rocm $CONTAINER  ./vAdd_ompGPU

