# Running MPI and HIP in Odo 

- Use apptainer to build the Apptainer container(Use the image location of the container that you built from frontier original docs)
- Run the command `apptainer build App_vAdd_mpi_hip.sif vAdd_mpi_hip.def` to create a container image file
- Run the command `sbatch submit.sbatch` if you are outside compute node 
- Run the command `sh submit.sbatch` if you have already allocated compute node and are inside it
- Run the command `make` if you want to check the code in your machine; (you need to have modules loaded and you should have cray compiler)

## Expected Output Format
```
buffer_size = 268435456
MPI 00 - HWT 047 - GPU 0 - Node login1
Test passed.
Result                 = 1.0000000000000000
Relative difference    = 0.0000000000000000
Tolerance              = 0.0000000000000100
Array buffer size (MB) = 256.00
Elapsed time (s)       = 0.0012497
```
