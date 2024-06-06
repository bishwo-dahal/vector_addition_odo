#include <hip/hip_runtime.h>
#include "vAdd_gpu.h"

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
__global__ void add_vectors(double *a, double *b, double *c, long long int n){

    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < n) c[id] = a[id] + b[id];
}

void vec_add_on_gpu(int rank_id, int omp_id, int hwt, char *node, double *sub_a, double *sub_b, double *sub_c, size_t omp_buff_size, long long int chunk){

    // Find number of devices
    int num_devices = 0;
    hipErrorCheck( hipGetDeviceCount(&num_devices) );

    // Set device based on MPI rank and find Bus ID
    int gpu_id = rank_id % num_devices;
    char bus_id[64];
    hipErrorCheck( hipSetDevice(gpu_id) );
    hipErrorCheck( hipDeviceGetPCIBusId(bus_id, 64, gpu_id) );

    printf("MPI %02d - OMPID %02d - HWT %03d - GPU %d (BUS ID %s) - Node %s\n", rank_id, omp_id, hwt, gpu_id, bus_id, node);
    fflush(stdout);

    double *d_sub_a, *d_sub_b, *d_sub_c;

    hipErrorCheck( hipMalloc(&d_sub_a, omp_buff_size) );
    hipErrorCheck( hipMalloc(&d_sub_b, omp_buff_size) );
    hipErrorCheck( hipMalloc(&d_sub_c, omp_buff_size) );

    hipErrorCheck( hipMemcpy(d_sub_a, sub_a, omp_buff_size, hipMemcpyHostToDevice) );
    hipErrorCheck( hipMemcpy(d_sub_b, sub_b, omp_buff_size, hipMemcpyHostToDevice) );

    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(chunk) / thr_per_blk );

    hipLaunchKernelGGL((add_vectors), dim3(blk_in_grid), dim3(thr_per_blk), 0, 0, d_sub_a, d_sub_b, d_sub_c, chunk);
    hipErrorCheck( hipDeviceSynchronize() );

    hipErrorCheck( hipMemcpy(sub_c, d_sub_c, omp_buff_size, hipMemcpyDeviceToHost) );

    hipErrorCheck( hipFree(d_sub_a) );
    hipErrorCheck( hipFree(d_sub_b) );
    hipErrorCheck( hipFree(d_sub_c) );

}
