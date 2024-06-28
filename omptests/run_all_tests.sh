#!/bin/bash



echo "Running all the OMP TESTS"



cd hip_ompCPU/ 
sh submit.sbatch
make clean

cd ../hip_ompCPU_separateFiles/ 
sh submit.sbatch
make clean

cd ../mpi_ompCPU/ 
sh submit.sbatch
make clean

cd ../mpi_ompCPU_hip/ 
sh submit.sbatch
make clean

cd ../mpi_ompGPU/ 
sh submit.sbatch
make clean

cd ../ompCPU/ 
sh submit.sbatch
make clean

cd ../ompGPU/ 
sh submit.sbatch
make clean

echo "Completed OMP TESTS"
