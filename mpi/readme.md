#Running MPI in Odo using g++

- Use apptainer to build the Apptainer container(Use the image location of the container that you built from frontier original docs)
- Run the command `apptainer build App_vAdd.sif vAdd.def` to create a container image file
- Run the command `sbatch submit.sbatch` if you are outside compute node 
- Run the command `sh submit.sbatch` if you have already allocated compute node and are inside it
- Run the command `make` if you want to check the code in your machine; (you need to have modules loaded and you should have cray compiler)

