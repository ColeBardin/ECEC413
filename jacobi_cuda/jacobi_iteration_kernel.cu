#include "jacobi_iteration.h"

/* FIXME: Write the device kernels to solve the Jacobi iterations */


__global__ void jacobi_iteration_kernel_naive(float *A, float *B, float *src, float *dst, double *ssd)
{
    __shared__ double ssds[THREAD_BLOCK_SIZE];
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int i;
    double sum;
    
    if(row < NUM_ROWS)
    {
        //printf("Thread %d\n", row);
        sum = -A[row * NUM_COLUMNS + row] * src[row];
        for(i = 0; i < NUM_COLUMNS; i++)
        {
            sum += A[row * NUM_COLUMNS + i] * src[i];
        }

        dst[row] = (B[row] - sum) / A[row * NUM_COLUMNS + row];

        ssds[threadIdx.y] = (src[row] - dst[row]) * (src[row] - dst[row]);
    }

    __syncthreads();

    i = THREAD_BLOCK_SIZE / 2;
    while(i > 0)
    {
        if(row < NUM_ROWS && threadIdx.y < i)
        {
            ssds[threadIdx.y] += ssds[threadIdx.y + i];
        }
        __syncthreads();
        i /= 2;
    }
    if(threadIdx.y == 0) atomicAdd(ssd, ssds[0]);
    return;
}

__global__ void jacobi_iteration_kernel_optimized(float *A, float *B, float *src, float *dst, double *ssd)
{
    __shared__ double ssds[THREAD_BLOCK_SIZE];
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int i;
    double sum;
    
    if(col < NUM_COLUMNS)
    {
        sum = -A[col * NUM_ROWS + col] * src[col];
        for(i = 0; i < NUM_ROWS; i++)
        {
            sum += A[i * NUM_ROWS + col] * src[i];
        }

        dst[col] = (B[col] - sum) / A[col * NUM_ROWS + col];

        ssds[threadIdx.x] = (src[col] - dst[col]) * (src[col] - dst[col]);
    }

    __syncthreads();

    i = THREAD_BLOCK_SIZE / 2;
    while(i > 0)
    {
        if(col < NUM_COLUMNS && threadIdx.x < i)
        {
            ssds[threadIdx.x] += ssds[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }
    if(threadIdx.x == 0) atomicAdd(ssd, ssds[0]);
    return;
}

