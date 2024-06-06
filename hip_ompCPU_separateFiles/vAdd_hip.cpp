#include <hip/hip_runtime.h>

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



void gpu_vector_add(long long int n, double* a, double* b, double *c, size_t buff_size, double toler){


    double *d_A, *d_B, *d_C;
    hipErrorCheck( hipMalloc(&d_A, buff_size) );
    hipErrorCheck( hipMalloc(&d_B, buff_size) );
    hipErrorCheck( hipMalloc(&d_C, buff_size) );

    hipErrorCheck( hipMemcpy(d_A, a, buff_size, hipMemcpyHostToDevice) );
    hipErrorCheck( hipMemcpy(d_B, b, buff_size, hipMemcpyHostToDevice) );

    hipEvent_t start_gpu, end_gpu;
    hipErrorCheck( hipEventCreate(&start_gpu) );
    hipErrorCheck( hipEventCreate(&end_gpu) );

    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(n) / thr_per_blk );

    hipErrorCheck( hipEventRecord(start_gpu, NULL) );

    hipLaunchKernelGGL((add_vectors), dim3(blk_in_grid), dim3(thr_per_blk), 0, 0, d_A, d_B, d_C, n);

    hipErrorCheck( hipEventRecord(end_gpu, NULL) );
    hipErrorCheck( hipEventSynchronize(end_gpu) );
    float milliseconds = 0.0;
    hipErrorCheck( hipEventElapsedTime(&milliseconds, start_gpu, end_gpu) );
    double elapsed_time_gpu = milliseconds / 1000.0;

    hipErrorCheck( hipMemcpy(c, d_C, buff_size, hipMemcpyDeviceToHost) );

    double sum_gpu = 0.0;
    for(int i=0; i<n; i++){
        sum_gpu = sum_gpu + c[i];
    }

    double result_gpu = sum_gpu / (double)n;
    double relative_difference_gpu = fabs( (result_gpu - 1.0) / 1.0 );

    if(relative_difference_gpu > toler){
        printf("Test failed! (GPU)\n");
    }
    else{
        printf("Test passed. (GPU)\n");
    }

    printf("Result (GPU)              = %.16f\n" , result_gpu);
    printf("Relative difference (GPU) = %.16f\n" , relative_difference_gpu);
    printf("Elapsed time (s; GPU)     = %.6f\n"  , elapsed_time_gpu);

    hipErrorCheck( hipFree(d_A) );
    hipErrorCheck( hipFree(d_B) );
    hipErrorCheck( hipFree(d_C) );

}
