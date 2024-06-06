#!/bin/bash
#SBATCH -A trn025
#SBATCH -N2
#SBATCH -J Vector addition on cpu(OMP)
#SBATCH -t 15:00
#SBATCH -o /res/%x_%j.out
#SBATCH -e /res/%x_%j.err


module load cce
module load amd-mixed/5.3.0
module load cray-mpich/8.1.23

export MPICH_GPU_SUPPORT_ENABLED=1

srun -N2 apptainer exec App_vectorAdditionOmpCpu.sif


