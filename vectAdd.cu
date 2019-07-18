#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

//for random intialize
#include <stdlib.h>
#include <time.h>

//for memset
#include <cstring>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char * file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line );
        if (abort) exit(code);
    }
}

__global__ void sum_array_gpu( int* a, int* b, int* c, int* result, int size)
{
    int gid = blockIdx.x *blockDim.x +threadIdx.x;
    
    if(gid < size)
    {
        result[gid] = a[gid] + b[gid] + c[gid];
        
    }
}

void sum_array_cpu( int* a, int* b, int* c, int* result, int size)
{
    
    for (int i=0; i < size; i++)
    {
        result[i] = a[i] + b[i] + c[i];
       
    }
}

void compare_arrays (int* gpu, int* cpu, int size){
    for ( int i = 0; i < size ; i++){
        if(gpu[i]!= cpu[i]){

            printf("Arrays are different \n");
            return;
        }
    }
    printf("Arrays are same \n");
}
 
int main()
{
    int size = pow(2,22);
    int block_size = 512;
    int NO_BYTES = size * sizeof(int);

    // Allocate memory in Host
    int* h_a, *h_b, *h_c, *gpu_results, *cpu_results;
    h_a = (int*)malloc(NO_BYTES);
    h_b = (int*)malloc(NO_BYTES);
    h_c = (int*)malloc(NO_BYTES);
    cpu_results = (int*)malloc(NO_BYTES);
    gpu_results = (int*)malloc(NO_BYTES);

    time_t t;
    srand((unsigned)time(&t));

    // Initialise random values for the array

    for (int i=0; i <size; i++)
    {
        h_a[i] = (int)(rand() & 0xff);
    }
    for (int i=0; i <size; i++)
    {
        h_b[i] = (int)(rand() & 0xff);
    }
    for (int i=0; i <size; i++)
    {
        h_c[i] = (int)(rand() & 0xff);
    }

    memset(gpu_results,0,NO_BYTES);
    memset(cpu_results,0,NO_BYTES);

    //Summation in CPU
    clock_t cpu_start, cpu_end;
    cpu_start = clock();
    sum_array_cpu(h_a, h_b, h_c, cpu_results, size);
    cpu_end = clock();


    

    // Allocate memory in device

    int* d_a, *d_b, *d_c, *d_result;
    gpuErrchk(cudaMalloc((int**)&d_a,NO_BYTES));
    gpuErrchk(cudaMalloc((int**)&d_b,NO_BYTES));
    gpuErrchk(cudaMalloc((int**)&d_c,NO_BYTES));
    gpuErrchk(cudaMalloc((int**)&d_result,NO_BYTES));

    clock_t htod_start, htod_end;
    htod_start = clock();
    // Transfer the data from host to device
    gpuErrchk(cudaMemcpy(d_a, h_a, NO_BYTES, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, h_b, NO_BYTES, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_c, h_c, NO_BYTES, cudaMemcpyHostToDevice));
    htod_end = clock();

    // Designing grid and block size
    dim3 block(block_size);
    dim3 grid((size/block.x)+1);
 
    // Launch kernel function
    clock_t gpu_start, gpu_end;
    gpu_start = clock();
    sum_array_gpu << < grid, block >> > (d_a, d_b, d_c, d_result, size);
    cudaDeviceSynchronize();
    gpu_end = clock();

    clock_t dtoh_start, dtoh_end;
    dtoh_start = clock();
    gpuErrchk(cudaMemcpy(gpu_results, d_result, NO_BYTES, cudaMemcpyDeviceToHost));
    dtoh_end = clock();

    //compare the arrays
    compare_arrays(gpu_results,cpu_results, size);
    
    printf("Sum array CPU execution time : %4.6f \n",
            (double)((double)(cpu_end - cpu_start)/ CLOCKS_PER_SEC));

    printf("Sum array GPU execution time : %4.6f \n",
            (double)((double)(gpu_end - gpu_start)/ CLOCKS_PER_SEC));
            
    printf("htod mem transfer time : %4.6f \n",
            (double)((double)(htod_end - htod_start)/ CLOCKS_PER_SEC));
            
    printf("dtoh mem transfer time : %4.6f \n",
            (double)((double)(dtoh_end - dtoh_start)/ CLOCKS_PER_SEC));

    printf("Sum array GPU total execution time : %4.6f \n",
            (double)((double)(dtoh_end - htod_start)/ CLOCKS_PER_SEC));


    cudaFree(d_result);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);


    free(gpu_results);
    free(h_a);
    free(h_b);
    free(h_c);


    cudaDeviceReset(); 
}