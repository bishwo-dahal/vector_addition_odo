#!/bin/bash
#SBATCH -A trn025
#SBATCH -N2
#SBATCH -J vector_addition_mpi_omp_CPU
#SBATCH -o res/%x_%j.out
#SBATCH -e res/%x_%j.err
#SBATCH -t 3:00



module load cce
module load amd-mixed/5.3.0
module load cray-mpich/8.1.23

export MPICH_GPU_SUPPORT_ENABLED=1

srun -N2 apptainer exec App_vAdd_mpi_ompCPU.sif /vAdd_mpi_ompCPU

