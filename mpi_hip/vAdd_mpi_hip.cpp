#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <sched.h>
#include <mpi.h>
#include <hip/hip_runtime.h>

/* ---------------------------------------------------------------------------------
Macro for checking errors in HIP API calls
----------------------------------------------------------------------------------*/
#define hipErrorCheck(call)                                                                 \
do{                                                                                         \
    hipError_t hipErr = call;                                                               \
    if(hipSuccess != hipErr){                                                               \
        printf("HIP Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(hipErr)); \
        exit(0);                                                                            \
    }                                                                                       \
}while(0)

/* ---------------------------------------------------------------------------------
Vector addition kernel
----------------------------------------------------------------------------------*/
__global__ void add_vectors(double *a, double *b, double *c, int n){

    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < n) c[id] = a[id] + b[id];
}

/* ---------------------------------------------------------------------------------
Main program
----------------------------------------------------------------------------------*/
int main(int argc, char *argv[]){

    /* MPI initialization ----------------------------------------- */

    MPI_Init(&argc, &argv);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char name[MPI_MAX_PROCESSOR_NAME];
    int result_length;
    MPI_Get_processor_name(name, &result_length);

    int hwthread = sched_getcpu();

    int num_devices = 0;
    hipErrorCheck( hipGetDeviceCount(&num_devices) );
    
    int gpu_id = rank % num_devices;
    hipErrorCheck( hipSetDevice(gpu_id) );

    // Array length
    long long int N = 32*1024*1024;

    double tolerance = 1.0e-14;

    size_t buffer_size = N * sizeof(double);
    printf("buffer_size = %zu\n", buffer_size);

    double *A = (double*)malloc(buffer_size);
    double *B = (double*)malloc(buffer_size);
    double *C = (double*)malloc(buffer_size);

    printf("MPI %02d - HWT %03d - GPU %d - Node %s\n", rank, hwthread, gpu_id, name);
    fflush(stdout);

    if(rank == 0){
        for(int i=0; i<N; i++){
            double random_value = (double)rand()/(double)RAND_MAX;
            A[i] = sin(random_value) * sin(random_value);
            B[i] = cos(random_value) * cos(random_value);
            C[i] = 0.0;
        }
    }

    int chunk_size = 0;
    if( (N % size) != 0){
        printf("N must be evenly divisible by size. Exiting...\n");
        MPI_Finalize();
        exit(-1);
    }
    else{
        chunk_size = N / size;
    }

    size_t sub_buffer_size = chunk_size * sizeof(double);
    double *sub_A = (double*)malloc(sub_buffer_size);
    double *sub_B = (double*)malloc(sub_buffer_size);
    double *sub_C = (double*)malloc(sub_buffer_size);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatter(A, chunk_size, MPI_DOUBLE, sub_A, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
    MPI_Scatter(B, chunk_size, MPI_DOUBLE, sub_B, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double *d_sub_A, *d_sub_B, *d_sub_C;
    hipErrorCheck( hipMalloc(&d_sub_A, sub_buffer_size) );
    hipErrorCheck( hipMalloc(&d_sub_B, sub_buffer_size) );
    hipErrorCheck( hipMalloc(&d_sub_C, sub_buffer_size) );

    hipErrorCheck( hipMemcpy(d_sub_A, sub_A, sub_buffer_size, hipMemcpyHostToDevice) );
    hipErrorCheck( hipMemcpy(d_sub_B, sub_B, sub_buffer_size, hipMemcpyHostToDevice) );
    hipErrorCheck( hipMemcpy(d_sub_C, sub_C, sub_buffer_size, hipMemcpyHostToDevice) );

    double start, end, elapsed_time;
    start = MPI_Wtime();

    // Set execution configuration parameters
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(chunk_size) / thr_per_blk );

    hipLaunchKernelGGL((add_vectors), dim3(blk_in_grid), dim3(thr_per_blk), 0, 0, d_sub_A, d_sub_B, d_sub_C, chunk_size);
    hipErrorCheck( hipDeviceSynchronize() );

    end = MPI_Wtime();
    elapsed_time = end - start;

    hipErrorCheck( hipMemcpy(sub_C, d_sub_C, sub_buffer_size, hipMemcpyDeviceToHost) );

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(sub_C, chunk_size, MPI_DOUBLE, C, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(rank == 0){

        double sum = 0.0;
        for(int i=0; i<N; i++){
            sum = sum + C[i];
        }

        double result = sum / (double)N;
        double relative_difference = fabs( (result - 1.0) / 1.0 );

        if(relative_difference > tolerance){
            printf("Test failed!\n");
        }
        else{
            printf("Test passed.\n");
        }

        printf("Result                 = %.16f\n", result);
        printf("Relative difference    = %.16f\n", relative_difference);
        printf("Tolerance              = %.16f\n", tolerance);
        printf("Array buffer size (MB) = %3.2f\n", (double)buffer_size / (1024.0*1024.0) );
        printf("Elapsed time (s)       = %9.7f\n", elapsed_time);
    }

    hipErrorCheck( hipFree(d_sub_A) );
    hipErrorCheck( hipFree(d_sub_B) );
    hipErrorCheck( hipFree(d_sub_C) );

    free(A);
    free(B);
    free(C);
    free(sub_A);
    free(sub_B);
    free(sub_C);

    MPI_Finalize();

    return 0;
}
