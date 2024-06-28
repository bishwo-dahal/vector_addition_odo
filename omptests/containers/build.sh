#!/bin/bash


export CONTAINERS=../containers

export APPTAINER=apptainer

# -e is to make sure the bash script exits as soon as any line in the script fails
# -x prints all the command that is being executed on current line
set -ex

# The main file for building all the sif files
$APPTAINER build -F opensusempich342rocm571.sif opensusempich342rocm571.def



#$APPTAINER build -F  vAdd_hip_ompCPU.sif vAdd_hip_ompCPU.def
#$APPTAINER build -F vAdd_hip_ompCPU_separateFiles.sif vAdd_hip_ompCPU_separateFiles.def
#$APPTAINER build -F vAdd_mpi_ompCPU.sif vAdd_mpi_ompCPU.def
#$APPTAINER build -F vAdd_mpi_ompCPU_hip.sif vAdd_mpi_ompCPU_hip.def
#$APPTAINER build -F vAdd_mpi_ompGPU.sif vAdd_mpi_ompGPU.def
#$APPTAINER build -F vAdd_ompCPU.sif vAdd_ompCPU.def
#$APPTAINER build -F vAdd_ompGPU.sif vAdd_ompGPU.def



