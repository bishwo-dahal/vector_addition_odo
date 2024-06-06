#include <stdio.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include <omp.h>

/* =================================================================================
Macro for checking errors in HIP API calls
================================================================================= */
#define hipErrorCheck(call)                                                                 \
do{                                                                                         \
    hipError_t hipErr = call;                                                               \
    if(hipSuccess != hipErr){                                                               \
        printf("HIP Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(hipErr)); \
        exit(0);                                                                            \
    }                                                                                       \
}while(0)

/* =================================================================================
Vector addition kernel
================================================================================= */
__global__ void add_vectors(double *a, double *b, double *c, int n){
    
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < n) c[id] = a[id] + b[id];
}

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


    /* ---------- GPU Calculation and Correctness Check ---------- */

    double *d_A, *d_B, *d_C;
    hipErrorCheck( hipMalloc(&d_A, buffer_size) );
    hipErrorCheck( hipMalloc(&d_B, buffer_size) );
    hipErrorCheck( hipMalloc(&d_C, buffer_size) );

    hipErrorCheck( hipMemcpy(d_A, A, buffer_size, hipMemcpyHostToDevice) );
    hipErrorCheck( hipMemcpy(d_B, B, buffer_size, hipMemcpyHostToDevice) );

    hipEvent_t start_gpu, end_gpu;
    hipErrorCheck( hipEventCreate(&start_gpu) );
    hipErrorCheck( hipEventCreate(&end_gpu) );

    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(N) / thr_per_blk );

    hipErrorCheck( hipEventRecord(start_gpu, NULL) );

    hipLaunchKernelGGL((add_vectors), dim3(blk_in_grid), dim3(thr_per_blk), 0, 0, d_A, d_B, d_C, N);

    hipErrorCheck( hipEventRecord(end_gpu, NULL) );
    hipErrorCheck( hipEventSynchronize(end_gpu) );
    float milliseconds = 0.0;
    hipErrorCheck( hipEventElapsedTime(&milliseconds, start_gpu, end_gpu) ); 
    double elapsed_time_gpu = milliseconds / 1000.0;

    hipErrorCheck( hipMemcpy(C, d_C, buffer_size, hipMemcpyDeviceToHost) );

    double sum_gpu = 0.0;
    for(int i=0; i<N; i++){
        sum_gpu = sum_gpu + C[i];
    }

    double result_gpu = sum_gpu / (double)N;
    double relative_difference_gpu = fabs( (result_gpu - 1.0) / 1.0 );

    if(relative_difference_gpu > tolerance){
        printf("Test failed! (GPU)\n");
    }
    else{
        printf("Test passed. (GPU)\n");
    }

    /* ---------- Print Results ---------- */

    printf("Array buffer size         = %zu\n"   , buffer_size);
    printf("Tolerance                 = %.16f\n" , tolerance);
    printf("Result (CPU)              = %.16f\n" , result_cpu);
    printf("Relative difference (CPU) = %.16f\n" , relative_difference_cpu);
    printf("Elapsed time (s; CPU)     = %.6f\n"  , elapsed_time_cpu); 
    printf("Result (GPU)              = %.16f\n" , result_gpu);
    printf("Relative difference (GPU) = %.16f\n" , relative_difference_gpu);
    printf("Elapsed time (s; GPU)     = %.6f\n"  , elapsed_time_gpu);

    /* ---------- Cleanup ---------- */

    hipErrorCheck( hipFree(d_A) );
    hipErrorCheck( hipFree(d_B) );
    hipErrorCheck( hipFree(d_C) );

    free(A);
    free(B);
    free(C);
    free(C_cpu);

    return 0;
}
