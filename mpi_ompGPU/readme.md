# Running MPI in Odo using g++

### MUST LOAD amd-mixed module to run on your local node/ compute node

- Use apptainer to build the Apptainer container(Use the image location of the container that you built from frontier original docs)
- Run the command `apptainer build App_vAdd_mpi_ompGPU.sif vAdd_vAdd_mpi_ompGPU.def` to create a container image file
- Run the command `sbatch submit.sbatch` if you are outside compute node 
- Run the command `sh submit.sbatch` if you have already allocated compute node and are inside it
- Run the command `make` if you want to check the code in your machine; (you need to have modules loaded and you should have cray compiler)

## Expected Output Format
```
MPI 00 - HWT 123 - GPU 0 - Node odo01 - num_devices 8 - device_id -10
Test passed.
Result                 = 1.0000000000000000
Tolerance              = 0.0000000000000100
Array buffer size (MB) = 256.00
Elapsed time (s)       = 0.2992496
```
