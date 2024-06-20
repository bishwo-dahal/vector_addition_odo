# Running HIP in Odo using hipcc

- Use apptainer to build the Apptainer container(Use the image location of the container that you built from frontier original docs)
- Run the command `apptainer build App_vAdd_hip.sif vAdd_hip.def` to create a container image file
- Run the command `sbatch submit.sbatch` if you are outside compute node 
- Run the command `sh submit.sbatch` if you have already allocated compute node and are inside it
- Run the command `make` if you want to check the code in your machine; (you need to have modules loaded and you should have cray compiler)

## Expected Output Format
```
Test passed.
Result              = 1.0000000000000000
Relative difference = 0.0000000000000000
Tolerance           = 0.0000000000000100
Array buffer size   = 2147483648
Elapsed time (s)    = 0.005825

```
