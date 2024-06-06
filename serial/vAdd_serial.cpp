#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

int main(int argc, char *argv[]){

    // Array length
    long long int N = 256*1024*1024;

    double tolerance = 1.0e-14;

    size_t buffer_size = N * sizeof(double);

    double *A = (double*)malloc(buffer_size);
    double *B = (double*)malloc(buffer_size);
    double *C = (double*)malloc(buffer_size);

    for(int i=0; i<N; i++){
        double random_value = (double)rand()/(double)RAND_MAX;
        A[i] = sin(random_value) * sin(random_value);
        B[i] = cos(random_value) * cos(random_value);
        C[i] = 0.0;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    for(int i=0; i<N; i++){
        C[i] = A[i] + B[i];
    } 

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t elapsed_time_us = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;

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

    printf("Result              = %.16f\n", result);
    printf("Relative difference = %.16f\n", relative_difference);
    printf("Tolerance           = %.16f\n", tolerance);
    printf("Array buffer size   = %zu\n", buffer_size);
    printf("Elapsed time (s)    = %10.7f\n", (double)elapsed_time_us/((double)(1024*1024)));

    return 0;
}
