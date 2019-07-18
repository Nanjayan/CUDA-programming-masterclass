#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void print_details( int* input)
{
    int tid = (threadIdx.z*blockDim.x*blockDim.y)+(threadIdx.y*blockDim.x)+threadIdx.x;
    int num_of_thread_in_a_block = blockDim.x * blockDim.y * blockDim.z;
    int block_offset = num_of_thread_in_a_block  * blockIdx.x;
    int num_of_threads_in_a_row = num_of_thread_in_a_block * gridDim.x;
    int row_offset = num_of_threads_in_a_row * blockIdx.y;
    int num_of_thread_in_xy = num_of_thread_in_a_block * gridDim.x * gridDim.y;
    int z_offset = num_of_thread_in_xy * blockIdx.z;

    int gid = tid + block_offset + row_offset + z_offset;   

    printf("tid : %d , gid : %d , value : %d \n", tid, gid, input[gid]);
}
 
int main()
{
    int size = 64;
    int byte_size = size * sizeof(int);

    int* h_input;
    h_input = (int*)malloc(byte_size);

    time_t t;
    srand((unsigned)time(&t));

    for (int i=0; i <size; i++)
    {
        h_input[i] = (int)(rand() & 0xff);
    }

    int* d_input;
    cudaMalloc((void**)&d_input,byte_size);

    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

 
    dim3 block(2,2,2);
    dim3 grid(2,2,2);
 
    print_details << < grid, block >> > (d_input);
    
    cudaDeviceSynchronize();

    cudaFree(d_input);
    free(h_input);


    cudaDeviceReset(); 
}
