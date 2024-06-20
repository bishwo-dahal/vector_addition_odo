#!/bin/bash
#SBATCH -A trn025
#SBATCH -N2
#SBATCH -J Vector_addition_on_cpu_OMP_HIP
#SBATCH -t 15:00
#SBATCH -o /res/%x_%j.out
#SBATCH -e /res/%x_%j.err


module load cce
module load amd-mixed/5.3.0
module load cray-mpich/8.1.23

export MPICH_GPU_SUPPORT_ENABLED=1

srun -N2 apptainer exec App_vAdd_hip_ompCPU.sif /vAdd_hip_ompCPU


