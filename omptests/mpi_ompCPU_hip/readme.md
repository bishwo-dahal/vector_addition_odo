# Running MPI, HIP and OMP in Odo 

- Use apptainer to build the Apptainer container(Use the image location of the container that you built from frontier original docs)
- Run the command `apptainer build App_vAdd_mpi_ompCPU_hip.sif vAdd_mpi_ompCPU_hip.def` to create a container image file
- Run the command `sbatch submit.sbatch` if you are outside compute node 
- Run the command `sh submit.sbatch` if you have already allocated compute node and are inside it
- Run the command `make` if you want to check the code in your machine; (you need to have modules loaded and you should have cray compiler)

## Expected Output Format
```
MPI 00 - OMPID 100 - HWT 008 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 90 - HWT 114 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 67 - HWT 078 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 04 - HWT 060 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 46 - HWT 013 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 34 - HWT 079 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 75 - HWT 073 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 104 - HWT 076 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 22 - HWT 116 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 69 - HWT 053 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 86 - HWT 115 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 127 - HWT 055 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 33 - HWT 009 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 109 - HWT 077 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 91 - HWT 055 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 59 - HWT 121 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 03 - HWT 062 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 16 - HWT 057 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 43 - HWT 113 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 02 - HWT 063 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 103 - HWT 118 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 82 - HWT 117 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 116 - HWT 075 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 42 - HWT 117 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 40 - HWT 049 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 19 - HWT 052 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 78 - HWT 014 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 49 - HWT 055 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 88 - HWT 119 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 23 - HWT 049 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 83 - HWT 115 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 36 - HWT 015 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 52 - HWT 123 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 07 - HWT 126 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 70 - HWT 052 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 71 - HWT 051 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 80 - HWT 013 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 48 - HWT 056 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 25 - HWT 124 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 68 - HWT 048 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 89 - HWT 116 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 54 - HWT 062 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 08 - HWT 059 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 32 - HWT 010 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 108 - HWT 077 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 17 - HWT 050 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 27 - HWT 124 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 107 - HWT 076 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 12 - HWT 058 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 60 - HWT 061 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 81 - HWT 113 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 28 - HWT 122 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 57 - HWT 121 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 26 - HWT 124 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 14 - HWT 125 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 120 - HWT 072 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 119 - HWT 075 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 56 - HWT 121 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 76 - HWT 014 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 84 - HWT 112 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 39 - HWT 015 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 112 - HWT 011 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 01 - HWT 124 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 96 - HWT 012 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 06 - HWT 124 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 24 - HWT 124 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 122 - HWT 008 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 41 - HWT 119 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 101 - HWT 115 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 58 - HWT 057 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 66 - HWT 049 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 30 - HWT 057 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 65 - HWT 050 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 77 - HWT 078 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 15 - HWT 120 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 126 - HWT 014 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 10 - HWT 059 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 53 - HWT 123 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 87 - HWT 113 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 106 - HWT 076 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 05 - HWT 060 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 64 - HWT 119 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 97 - HWT 074 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 111 - HWT 077 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 20 - HWT 050 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 102 - HWT 049 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 50 - HWT 055 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 21 - HWT 048 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 29 - HWT 122 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 55 - HWT 062 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 11 - HWT 126 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 38 - HWT 015 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 00 - HWT 127 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 85 - HWT 112 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 61 - HWT 051 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 31 - HWT 121 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 51 - HWT 117 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 13 - HWT 058 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 74 - HWT 073 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 37 - HWT 079 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 123 - HWT 008 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 114 - HWT 011 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 63 - HWT 125 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 121 - HWT 072 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 62 - HWT 118 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 09 - HWT 124 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 44 - HWT 013 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 45 - HWT 013 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 125 - HWT 119 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 105 - HWT 012 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 118 - HWT 075 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 98 - HWT 074 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 110 - HWT 077 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 18 - HWT 050 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 73 - HWT 078 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 117 - HWT 075 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 72 - HWT 073 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 99 - HWT 063 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 113 - HWT 011 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 95 - HWT 119 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 124 - HWT 014 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 79 - HWT 009 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 92 - HWT 013 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 94 - HWT 052 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 115 - HWT 008 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 93 - HWT 116 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 47 - HWT 079 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
MPI 00 - OMPID 35 - HWT 074 - GPU 0 (BUS ID 0000:c9:00.0) - Node login1
Test passed.
Result                 = 1.0000000000000000
Tolerance              = 0.0000000000000100
Array buffer size (MB) = 256.00
```
