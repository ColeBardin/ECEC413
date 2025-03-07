/* GPU kernel to estimate integral of the provided function using the trapezoidal rule. */

/* Device function which implements the function. Device functions can be called from within other __device__ functions or __global__ functions (the kernel), but cannot be called from the host. */ 

__device__ float fd(float x) 
{
     return sqrtf((1 + x*x)/(1 + x*x*x*x));
}

/* Kernel function */
__global__ void trap_kernel(float a, float b, int n, float h) 
{
     
}
