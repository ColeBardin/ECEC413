/* Reference code implementing numerical integration.
 *
 * Build and execute as follows: 
        make clean && make 
        ./trap a b n 

 * Author: Naga Kandasamy
 * Date modified: February 28, 2025

 * Student name(s): Cole Bardin 
 * Date modified: 3/7/25
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

/* Include the kernel code */
#include "trap_kernel.cu"

double compute_on_device(float, float, int, float);
extern "C" double compute_gold(float, float, int, float);

int main(int argc, char **argv) 
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s a b n\n", argv[0]);
        fprintf(stderr, "a: Start limit. \n");
        fprintf(stderr, "b: end limit\n");
        fprintf(stderr, "n: number of trapezoids\n");
        exit(EXIT_FAILURE);
    }

    int a = atoi(argv[1]); /* Left limit */
    int b = atoi(argv[2]); /* Right limit */
    int n = atoi(argv[3]); /* Number of trapezoids */ 

    float h = (b-a)/(float)n; // Height of each trapezoid  
	printf("Number of trapezoids = %d\n", n);
    printf("Height of each trapezoid = %f \n", h);

	struct timeval start, stop;

	gettimeofday(&start, NULL);
	double reference = compute_gold(a, b, n, h);
	gettimeofday(&stop, NULL);
    printf("Reference solution computed on the CPU = %f \n", reference);
	printf("Execution time CPU = %f s\n", (float)(stop.tv_sec - start.tv_sec\
				+ (stop.tv_usec - start.tv_usec)/(float)1000000));

	/* Write this function to complete the trapezoidal on the GPU. */
	gettimeofday(&start, NULL);
	double gpu_result = compute_on_device(a, b, n, h);
	gettimeofday(&stop, NULL);
	printf("Solution computed on the GPU = %f \n", gpu_result);
	printf("Execution time GPU= %f s\n", (float)(stop.tv_sec - start.tv_sec\
				+ (stop.tv_usec - start.tv_usec)/(float)1000000));
} 

/* Complete this function to perform the trapezoidal rule on the GPU. */
double compute_on_device(float a, float b, int n, float h)
{
    double integral;
    double *ret_on_device;

    cudaMalloc((void **)&ret_on_device, sizeof(double));
    cudaMemset(ret_on_device, 0.0f, sizeof(double));

    dim3 tb(TB_SZ);
    dim3 grid((n + TB_SZ - 1) / TB_SZ);
    trap_kernel<<< grid, tb >>>(a, b, n, h, ret_on_device);
    cudaDeviceSynchronize();

    cudaMemcpy(&integral, ret_on_device, sizeof(double), cudaMemcpyDeviceToHost);

    return integral;
}



