# Running HIP & OMP in Odo using hipcc

- Use apptainer to build the Apptainer container(Use the image location of the container that you built from frontier original docs)
- Run the command `apptainer build App_vAdd_hip_ompCPU.sif vAdd_hip_ompCPU.def` to create a container image file
- Run the command `sbatch submit.sbatch` if you are outside compute node 
- Run the command `sh submit.sbatch` if you have already allocated compute node and are inside it
- Run the command `make` if you want to check the code in your machine; (you need to have modules loaded and you should have cray compiler)

## Expected Output Format
```
Test passed (CPU).
Test passed. (GPU)
Array buffer size         = 2147483648
Tolerance                 = 0.0000000000000100
Result (CPU)              = 1.0000000000000000
Relative difference (CPU) = 0.0000000000000000
Elapsed time (s; CPU)     = 0.246709
Result (GPU)              = 1.0000000000000000
Relative difference (GPU) = 0.0000000000000000
Elapsed time (s; GPU)     = 0.005789
```
