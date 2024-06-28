#!/bin/bash
#SBATCH -A trn025
#SBATCH -N2
#SBATCH -J Vector_addition_on_cpu_OMP_HIP
#SBATCH -t 15:00
#SBATCH -o /res/%x_%j.out
#SBATCH -e /res/%x_%j.err


module load cce
module load amd-mixed/5.7.1
module load cray-mpich/8.1.23

export MPICH_GPU_SUPPORT_ENABLED=1

set -ex

export CONTAINER=../containers/opensusempich342rocm571.sif
export APPTAINER=apptainer

srun -N2 $APPTAINER exec --rocm  $CONTAINER  hipcc  -fopenmp -fopenmp-extensions -target x86_64-pc-linux-gnu  --offload-arch=gfx90a  ./vAdd_hip_ompCPU.cpp -o vAdd_hip_ompCPU
srun -N2 $APPTAINER exec $CONTAINERS/opensusempich342rocm571.sif ./vAdd_hip_ompCPU

