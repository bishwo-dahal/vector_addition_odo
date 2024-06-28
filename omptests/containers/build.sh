#!/bin/bash


export CONTAINERS=../containers

export APPTAINER=apptainer

# -e is to make sure the bash script exits as soon as any line in the script fails
# -x prints all the command that is being executed on current line
set -ex

# The main file for building all the sif files
$APPTAINER build -F opensusempich342rocm571.sif opensusempich342rocm571.def

# If you want to build separate sif files for all tests, do it here below
