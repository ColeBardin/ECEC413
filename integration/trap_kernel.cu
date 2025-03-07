/* GPU kernel to estimate integral of the provided function using the trapezoidal rule. */

/* Device function which implements the function. Device functions can be called from within other __device__ functions or __global__ functions (the kernel), but cannot be called from the host. */ 

#define TB_SZ 512

__device__ float fd(float x) 
{
     return sqrtf((1 + x*x)/(1 + x*x*x*x));
}

/* Kernel function */
__global__ void trap_kernel(float a, float b, int n, float h, double *ret) 
{
    __shared__ double int_per_thread[TB_SZ];
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;     
    size_t stride = blockDim.x * gridDim.x;
    double sum = 0.0f;
    unsigned int i = tid;
    float x;
    
    // Seperate statement for first and last value
    // using ternary operaton in while loop below double execution time
    if(i == 0 || i == n - 1)
    {
        x = a + h * i;
        sum += fd(x) / 2.0f;
        i += stride;
    }
    while(i < n)
    {
        x = a + h * i;
        sum += fd(x);
        i += stride;
    }

    sum *= h;
    int_per_thread[threadIdx.x] = sum;
    __syncthreads();

    i = TB_SZ / 2; 
    while(i > 0)
    {
        if(threadIdx.x < i)
        {
            int_per_thread[threadIdx.x] += int_per_thread[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    if(threadIdx.x == 0) atomicAdd(ret, int_per_thread[0]);
}

