#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "vAdd_hip.h"

/* =================================================================================
Main program
================================================================================= */
int main(int argc, char *argv[]){

    /* ---------- Initialization  ---------- */
    long long int N = 256*1024*1024;

    double tolerance = 1.0e-14;

    size_t buffer_size = N * sizeof(double);

    double *A     = (double*)malloc(buffer_size);
    double *B     = (double*)malloc(buffer_size);
    double *C     = (double*)malloc(buffer_size);
    double *C_cpu = (double*)malloc(buffer_size);

    for(int i=0; i<N; i++){
        double random_value = (double)rand()/(double)RAND_MAX;
        A[i]     = sin(random_value) * sin(random_value);
        B[i]     = cos(random_value) * cos(random_value);
        C[i]     = 0.0;
        C_cpu[i] = 0.0;
    }

    /* ---------- CPU Calculation and Correctness Check ---------- */

    double start_cpu, end_cpu;
    start_cpu = omp_get_wtime();

    #pragma omp parallel default(shared)
    {
    #pragma omp for
    for(int i=0; i<N; i++){
        C_cpu[i] = A[i] + B[i];
    }
    }

    end_cpu = omp_get_wtime();
    double elapsed_time_cpu = end_cpu - start_cpu;

    double sum_cpu = 0.0;
    for(int i=0; i<N; i++){
        sum_cpu = sum_cpu + C_cpu[i];
    }

    double result_cpu = sum_cpu / (double)N;
    double relative_difference_cpu = fabs( (result_cpu - 1.0) / 1.0 );

    if(relative_difference_cpu > tolerance){
        printf("Test failed (CPU)!\n");
    }
    else{
        printf("Test passed (CPU).\n");
    }


    /* ---------- Print Results ---------- */

    printf("Array buffer size         = %zu\n"   , buffer_size);
    printf("Tolerance                 = %.16f\n" , tolerance);
    printf("--------------------------------------------\n");
    printf("Result (CPU)              = %.16f\n" , result_cpu);
    printf("Relative difference (CPU) = %.16f\n" , relative_difference_cpu);
    printf("Elapsed time (s; CPU)     = %.6f\n"  , elapsed_time_cpu); 

    /* ---------- Perform Calculation on GPU ---------- */

    gpu_vector_add(N, A, B, C, buffer_size, tolerance);

    /* ---------- Cleanup ---------- */

    free(A);
    free(B);
    free(C);
    free(C_cpu);

    return 0;
}
