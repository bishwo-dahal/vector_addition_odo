# Running OMP tests for ODO

- If you want to use `run_all_tests.sh` you have to allocate some nodes using `salloc -A trn025 -t 13:00 -N4`
- Run `./run_all_tests.sh` after allocating compute nodes
- Use apptainer to build the Apptainer container(Use the image location of the container that you built from frontier original docs)

- Run the command `sbatch submit.sbatch` if you are outside compute node

- Run the command `sh submit.sbatch` if you have already allocated compute node and are inside it
- Run the command `make` if you want to check the code in your machine; (you need to have modules loaded and you should have cray compiler)

- If you want to build a separate sif file for this specific example build using definition file from ../containers/*.def
- Run the command `apptainer build App_vAdd_ompGPU.sif vAdd_ompGPU.def` to create a container image file
