#!/bin/bash
#SBATCH -t 15:00
#SBATCH -A trn025
#SBATCH -N1
#SBATCH -J vector_addition_mpi
#SBATCH -o res/%x_%j.out
#SBATCH -e res/err_%x_%j.err


module load cce
module load   amd-mixed/5.3.0 

module load  cray-mpich/8.1.23

# This is to run GPU with MPI
export MPICH_GPU_SUPPORT_ENABLED=1

srun -N1 --gpus-per-task=5 apptainer exec App_vAdd_hip.sif  /vAdd_hip
