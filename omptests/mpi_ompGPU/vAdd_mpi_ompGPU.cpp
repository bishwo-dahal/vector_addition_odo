#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <sched.h>
#include <mpi.h>
#include <omp.h>

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

    long long int N = 32*1024*1024;
    size_t buffer_size = N * sizeof(double);

    double *A = (double*)malloc(buffer_size);
    double *B = (double*)malloc(buffer_size);
    double *C = (double*)malloc(buffer_size);

    double tolerance = 1.0e-14;

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

    MPI_Scatter(A, chunk_size, MPI_DOUBLE, sub_A, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
    MPI_Scatter(B, chunk_size, MPI_DOUBLE, sub_B, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int num_devices = omp_get_num_devices();
    int gpu_id = rank % num_devices; 
    omp_set_default_device(gpu_id);

    int device_id = omp_get_device_num();

    printf("MPI %02d - HWT %03d - GPU %d - Node %s - num_devices %d - device_id %d\n", rank, hwthread, gpu_id, name, num_devices, device_id);
    fflush(stdout);

    double start, end, elapsed_time;
    start = MPI_Wtime();

    #pragma omp target device(gpu_id) map(to:sub_A[:chunk_size],sub_B[:chunk_size]) map(tofrom:sub_C[:chunk_size])
    {
    #pragma omp teams distribute parallel for simd
    for(int i=0; i<chunk_size; i++){
        sub_C[i] = sub_A[i] + sub_B[i];
    }
    }

    end = MPI_Wtime();
    elapsed_time = end - start;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(sub_C, chunk_size, MPI_DOUBLE, C, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(rank == 0){

        double sum = 0.0;
        for(int i=0; i<N; i++){
            sum = sum + C[i];
        }

        double result = sum / (double)N;

        if( fabs(result - 1.0)  > tolerance){
            printf("Test failed!\n");
        }
        else{
            printf("Test passed.\n");
        }

        printf("Result                 = %.16f\n", result);
        printf("Tolerance              = %.16f\n", tolerance);
        printf("Array buffer size (MB) = %3.2f\n", (double)buffer_size / (1024.0 * 1024.0));
        printf("Elapsed time (s)       = %9.7f\n", elapsed_time);
    }

    free(A);
    free(B);
    free(C);
    free(sub_A);
    free(sub_B);
    free(sub_C);

    MPI_Finalize();

    return 0;
}
