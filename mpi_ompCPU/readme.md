#Running MPI in Odo using g++

- Use apptainer to build the Apptainer container(Use the image location of the container that you built from frontier original docs)
- Run the command `apptainer build App_vAdd_mpi_ompCPU.sif vAdd_mpi_ompCPU.def` to create a container image file
- Run the command `sbatch submit.sbatch` if you are outside compute node 
- Run the command `sh submit.sbatch` if you have already allocated compute node and are inside it
- Run the command `make` if you want to check the code in your machine; (you need to have modules loaded and you should have cray compiler)

## Expected Output format

```
MPI 00 OMP 00 - HWT 127 - Node login1
MPI 00 OMP 96 - HWT 059 - Node login1
MPI 00 OMP 65 - HWT 067 - Node login1
MPI 00 OMP 84 - HWT 115 - Node login1
MPI 00 OMP 19 - HWT 036 - Node login1
MPI 00 OMP 49 - HWT 071 - Node login1
MPI 00 OMP 125 - HWT 001 - Node login1
MPI 00 OMP 77 - HWT 015 - Node login1
MPI 00 OMP 45 - HWT 077 - Node login1
MPI 00 OMP 115 - HWT 041 - Node login1
MPI 00 OMP 122 - HWT 026 - Node login1
MPI 00 OMP 120 - HWT 062 - Node login1
MPI 00 OMP 34 - HWT 020 - Node login1
MPI 00 OMP 03 - HWT 090 - Node login1
MPI 00 OMP 76 - HWT 116 - Node login1
MPI 00 OMP 12 - HWT 052 - Node login1
MPI 00 OMP 68 - HWT 114 - Node login1
MPI 00 OMP 75 - HWT 100 - Node login1
MPI 00 OMP 108 - HWT 112 - Node login1
MPI 00 OMP 35 - HWT 101 - Node login1
MPI 00 OMP 50 - HWT 086 - Node login1
MPI 00 OMP 66 - HWT 081 - Node login1
MPI 00 OMP 06 - HWT 017 - Node login1
MPI 00 OMP 01 - HWT 050 - Node login1
MPI 00 OMP 92 - HWT 119 - Node login1
MPI 00 OMP 60 - HWT 113 - Node login1
MPI 00 OMP 27 - HWT 098 - Node login1
MPI 00 OMP 58 - HWT 023 - Node login1
MPI 00 OMP 11 - HWT 035 - Node login1
MPI 00 OMP 26 - HWT 083 - Node login1
MPI 00 OMP 126 - HWT 018 - Node login1
MPI 00 OMP 102 - HWT 022 - Node login1
MPI 00 OMP 118 - HWT 019 - Node login1
MPI 00 OMP 16 - HWT 121 - Node login1
MPI 00 OMP 80 - HWT 057 - Node login1
MPI 00 OMP 78 - HWT 088 - Node login1
MPI 00 OMP 69 - HWT 074 - Node login1
MPI 00 OMP 13 - HWT 079 - Node login1
MPI 00 OMP 112 - HWT 061 - Node login1
MPI 00 OMP 101 - HWT 007 - Node login1
MPI 00 OMP 09 - HWT 003 - Node login1
MPI 00 OMP 109 - HWT 006 - Node login1
MPI 00 OMP 02 - HWT 000 - Node login1
MPI 00 OMP 25 - HWT 004 - Node login1
MPI 00 OMP 42 - HWT 085 - Node login1
MPI 00 OMP 18 - HWT 082 - Node login1
MPI 00 OMP 105 - HWT 011 - Node login1
MPI 00 OMP 116 - HWT 054 - Node login1
MPI 00 OMP 90 - HWT 080 - Node login1
MPI 00 OMP 83 - HWT 107 - Node login1
MPI 00 OMP 79 - HWT 108 - Node login1
MPI 00 OMP 124 - HWT 049 - Node login1
MPI 00 OMP 17 - HWT 002 - Node login1
MPI 00 OMP 44 - HWT 055 - Node login1
MPI 00 OMP 117 - HWT 005 - Node login1
MPI 00 OMP 89 - HWT 064 - Node login1
MPI 00 OMP 33 - HWT 069 - Node login1
MPI 00 OMP 29 - HWT 075 - Node login1
MPI 00 OMP 46 - HWT 093 - Node login1
MPI 00 OMP 28 - HWT 053 - Node login1
MPI 00 OMP 100 - HWT 117 - Node login1
MPI 00 OMP 81 - HWT 066 - Node login1
MPI 00 OMP 110 - HWT 021 - Node login1
MPI 00 OMP 85 - HWT 014 - Node login1
MPI 00 OMP 36 - HWT 118 - Node login1
MPI 00 OMP 121 - HWT 008 - Node login1
MPI 00 OMP 98 - HWT 029 - Node login1
MPI 00 OMP 113 - HWT 009 - Node login1
MPI 00 OMP 52 - HWT 048 - Node login1
MPI 00 OMP 95 - HWT 046 - Node login1
MPI 00 OMP 106 - HWT 028 - Node login1
MPI 00 OMP 104 - HWT 060 - Node login1
MPI 00 OMP 70 - HWT 089 - Node login1
MPI 00 OMP 48 - HWT 125 - Node login1
MPI 00 OMP 87 - HWT 099 - Node login1
MPI 00 OMP 73 - HWT 068 - Node login1
MPI 00 OMP 57 - HWT 065 - Node login1
MPI 00 OMP 39 - HWT 044 - Node login1
MPI 00 OMP 62 - HWT 095 - Node login1
MPI 00 OMP 86 - HWT 031 - Node login1
MPI 00 OMP 51 - HWT 096 - Node login1
MPI 00 OMP 31 - HWT 043 - Node login1
MPI 00 OMP 21 - HWT 072 - Node login1
MPI 00 OMP 32 - HWT 123 - Node login1
MPI 00 OMP 20 - HWT 051 - Node login1
MPI 00 OMP 30 - HWT 092 - Node login1
MPI 00 OMP 47 - HWT 104 - Node login1
MPI 00 OMP 111 - HWT 037 - Node login1
MPI 00 OMP 54 - HWT 094 - Node login1
MPI 00 OMP 38 - HWT 091 - Node login1
MPI 00 OMP 61 - HWT 073 - Node login1
MPI 00 OMP 05 - HWT 010 - Node login1
MPI 00 OMP 67 - HWT 034 - Node login1
MPI 00 OMP 127 - HWT 032 - Node login1
MPI 00 OMP 94 - HWT 030 - Node login1
MPI 00 OMP 37 - HWT 076 - Node login1
MPI 00 OMP 97 - HWT 012 - Node login1
MPI 00 OMP 93 - HWT 013 - Node login1
MPI 00 OMP 99 - HWT 045 - Node login1
MPI 00 OMP 23 - HWT 109 - Node login1
MPI 00 OMP 123 - HWT 040 - Node login1
MPI 00 OMP 63 - HWT 106 - Node login1
MPI 00 OMP 107 - HWT 042 - Node login1
MPI 00 OMP 119 - HWT 038 - Node login1
MPI 00 OMP 43 - HWT 102 - Node login1
MPI 00 OMP 04 - HWT 103 - Node login1
MPI 00 OMP 59 - HWT 033 - Node login1
MPI 00 OMP 103 - HWT 039 - Node login1
MPI 00 OMP 91 - HWT 097 - Node login1
MPI 00 OMP 40 - HWT 124 - Node login1
MPI 00 OMP 15 - HWT 111 - Node login1
MPI 00 OMP 55 - HWT 105 - Node login1
MPI 00 OMP 08 - HWT 120 - Node login1
MPI 00 OMP 07 - HWT 110 - Node login1
MPI 00 OMP 41 - HWT 070 - Node login1
MPI 00 OMP 71 - HWT 047 - Node login1
MPI 00 OMP 53 - HWT 078 - Node login1
MPI 00 OMP 24 - HWT 122 - Node login1
MPI 00 OMP 74 - HWT 087 - Node login1
MPI 00 OMP 10 - HWT 016 - Node login1
MPI 00 OMP 72 - HWT 056 - Node login1
MPI 00 OMP 22 - HWT 024 - Node login1
MPI 00 OMP 64 - HWT 063 - Node login1
MPI 00 OMP 88 - HWT 058 - Node login1
MPI 00 OMP 14 - HWT 025 - Node login1
MPI 00 OMP 56 - HWT 126 - Node login1
MPI 00 OMP 114 - HWT 027 - Node login1
MPI 00 OMP 82 - HWT 084 - Node login1
Test passed.
Result                 = 1.0000000000000000
Tolerance              = 0.0000000000000100
Array buffer size (MB) = 256.00
Elapsed time (s)       = 0.0284461

```
