/* Host code for the Jacobi method of solving a system of linear equations 
 * by iteration.

 * Build as follows: make clean && make

 * Author: Naga Kandasamy
 * Date modified: March 6, 2025
 *
 * Student name(s); Cole Bardin
 * Date modified: 3/14/25
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "jacobi_iteration.h"

/* Include the kernel code */
#include "jacobi_iteration_kernel.cu"

/* Uncomment the line below if you want the code to spit out debug information. */ 
/* #define DEBUG */

int main(int argc, char **argv) 
{
	if (argc > 1) {
		printf("This program accepts no arguments\n");
		exit(EXIT_FAILURE);
	}

    matrix_t  A;                    /* N x N constant matrix */
	matrix_t  B;                    /* N x 1 b matrix */
	matrix_t reference_x;           /* Reference solution */ 
	matrix_t gpu_naive_solution_x;  /* Solution computed by naive kernel */
    matrix_t gpu_opt_solution_x;    /* Solution computed by optimized kernel */

	/* Initialize the random number generator */
	srand(time(NULL));

	/* Generate diagonally dominant matrix */ 
    printf("\nGenerating %d x %d system\n", MATRIX_SIZE, MATRIX_SIZE);
	A = create_diagonally_dominant_matrix(MATRIX_SIZE, MATRIX_SIZE);
	if (A.elements == NULL) {
        printf("Error creating matrix\n");
        exit(EXIT_FAILURE);
	}
	
    /* Create the other vectors */
    B = allocate_matrix_on_host(MATRIX_SIZE, 1, 1);
	reference_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);
	gpu_naive_solution_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);
    gpu_opt_solution_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);

#ifdef DEBUG
	print_matrix(A);
	print_matrix(B);
	print_matrix(reference_x);
#endif

	struct timeval start, stop;

    /* Compute Jacobi solution on CPU */
	printf("\nPerforming Jacobi iteration on the CPU\n");
	gettimeofday(&start, NULL);
    compute_gold(A, reference_x, B);
	gettimeofday(&stop, NULL);
    display_jacobi_solution(A, reference_x, B); /* Display statistics */
	printf("Execution time CPU = %f s\n", (float)(stop.tv_sec - start.tv_sec\
				+ (stop.tv_usec - start.tv_usec)/(float)1000000));
	
	/* Compute Jacobi solution on device. Solutions are returned 
       in gpu_naive_solution_x and gpu_opt_solution_x. */
    printf("\nPerforming Jacobi iteration on device\n");
	compute_on_device(A, gpu_naive_solution_x, gpu_opt_solution_x, B);
    
    free(A.elements); 
	free(B.elements); 
	free(reference_x.elements); 
	free(gpu_naive_solution_x.elements);
    free(gpu_opt_solution_x.elements);
	
    exit(EXIT_SUCCESS);
}


void compute_on_device(const matrix_t A, matrix_t gpu_naive_sol_x, 
                       matrix_t gpu_opt_sol_x, const matrix_t B)
{
	struct timeval start, stop;

    printf("Naive solution:\n");
	gettimeofday(&start, NULL);
    solve_cuda_naive(A, gpu_naive_sol_x, B);
	gettimeofday(&stop, NULL);
    display_jacobi_solution(A, gpu_naive_sol_x, B); /* Display statistics */
	printf("Execution time GPU Naive = %f s\n", (float)(stop.tv_sec - start.tv_sec\
				+ (stop.tv_usec - start.tv_usec)/(float)1000000));

    printf("Optimized solution:\n");
	gettimeofday(&start, NULL);
    solve_cuda_optimized(A, gpu_opt_sol_x, B);
	gettimeofday(&stop, NULL);
    display_jacobi_solution(A, gpu_opt_sol_x, B); 
	printf("Execution time GPU Optimized = %f s\n", (float)(stop.tv_sec - start.tv_sec\
				+ (stop.tv_usec - start.tv_usec)/(float)1000000));
    return;
}

void solve_cuda_naive(const matrix_t A, matrix_t x, const matrix_t B)
{
    int num_iter;
    bool done;
    matrix_t A_dev = allocate_matrix_on_device(A);
    matrix_t x_dev = allocate_matrix_on_device(x);
    matrix_t B_dev = allocate_matrix_on_device(B);

    copy_matrix_to_device(A_dev, A); 
    copy_matrix_to_device(x_dev, B); 
    copy_matrix_to_device(B_dev, B); 

    matrix_t new_x_dev = allocate_matrix_on_device(x);

    double mse;
    double ssd;
    double *ssd_dev;
    cudaMalloc((void **)&ssd_dev, sizeof(double));

    dim3 tb(1, THREAD_BLOCK_SIZE);
    dim3 grid(1, (NUM_ROWS + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE);

    float *src;
    float *dst;
    float *temp;

    src = x_dev.elements;
    dst = new_x_dev.elements;
    ssd = 0.0f;
    done = false;
    num_iter = 0;
    while(!done)
    {
        cudaMemset(ssd_dev, 0.0f, sizeof(double));

        jacobi_iteration_kernel_naive<<< grid, tb >>>(A_dev.elements, B_dev.elements, src, dst, ssd_dev);
        cudaDeviceSynchronize();

        num_iter++;
        cudaMemcpy(&ssd, ssd_dev, sizeof(double), cudaMemcpyDeviceToHost); 
        mse = sqrt(ssd);
        if(mse <= THRESHOLD || num_iter >= MAX_ITER) done = true;
        temp = src;
        src = dst;
        dst = temp;
    }

    //printf("\nConvergence achieved after %d iterations \n", num_iter);
    copy_matrix_from_device(x, x_dev);

    cudaFree(A_dev.elements);
    cudaFree(x_dev.elements);
    cudaFree(B_dev.elements);
    cudaFree(new_x_dev.elements);
    cudaFree(ssd_dev);
}

void solve_cuda_optimized(const matrix_t A, matrix_t x, const matrix_t B)
{
    int num_iter;
    bool done;

    matrix_t At = allocate_matrix_on_host(MATRIX_SIZE, MATRIX_SIZE, 0);
    matrix_transpose(A, At);

    matrix_t A_dev = allocate_matrix_on_device(A);
    matrix_t x_dev = allocate_matrix_on_device(x);
    matrix_t B_dev = allocate_matrix_on_device(B);

    copy_matrix_to_device(A_dev, At); 
    copy_matrix_to_device(x_dev, B); 
    copy_matrix_to_device(B_dev, B); 
    
    matrix_t new_x_dev = allocate_matrix_on_device(x);

    double mse;
    double ssd;
    double *ssd_dev;
    cudaMalloc((void **)&ssd_dev, sizeof(double));

    dim3 tb(THREAD_BLOCK_SIZE);
    dim3 grid((NUM_ROWS + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE);

    float *src;
    float *dst;
    float *temp;

    src = x_dev.elements;
    dst = new_x_dev.elements;
    done = false;
    num_iter = 0;
    while(!done)
    {
        cudaMemset(ssd_dev, 0.0f, sizeof(double));

        jacobi_iteration_kernel_optimized<<< grid, tb >>>(A_dev.elements, B_dev.elements, src, dst, ssd_dev);

        num_iter++;
        cudaMemcpy(&ssd, ssd_dev, sizeof(double), cudaMemcpyDeviceToHost); 
        mse = sqrt(ssd);
        if(mse <= THRESHOLD || num_iter >= MAX_ITER) done = true;
        temp = src;
        src = dst;
        dst = temp;
    }

    //printf("\nConvergence achieved after %d iterations \n", num_iter);
    copy_matrix_from_device(x, x_dev);

    cudaFree(A_dev.elements);
    cudaFree(x_dev.elements);
    cudaFree(B_dev.elements);
    cudaFree(new_x_dev.elements);
    free(At.elements);
}

/* Allocate matrix on the device of same size as M */
matrix_t allocate_matrix_on_device(const matrix_t M)
{
    matrix_t Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof(float);
    cudaMalloc((void **)&Mdevice.elements, size);
    return Mdevice;
}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
*/
matrix_t allocate_matrix_on_host(int num_rows, int num_columns, int init)
{	
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
		
	M.elements = (float *)malloc(size * sizeof(float));
	for (unsigned int i = 0; i < size; i++) {
		if (init == 0) 
            M.elements[i] = 0; 
		else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}
    
    return M;
}	

void matrix_transpose(const matrix_t src, matrix_t dst)
{
    int row, col;
    if(src.num_rows != dst.num_rows ||
       src.num_columns != dst.num_columns ||
       src.elements == NULL || dst.elements == NULL
      ) return;

    for(row = 0; row < src.num_rows; row++)
    {
        for(col = 0; col < src.num_columns; col++)
        {
            dst.elements[col * src.num_rows + row] = src.elements[row * src.num_columns + col];
        }
    }
}

/* Copy matrix to device */
void copy_matrix_to_device(matrix_t Mdevice, const matrix_t Mhost)
{
    int size = Mhost.num_rows * Mhost.num_columns * sizeof(float);
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
    return;
}

/* Copy matrix from device to host */
void copy_matrix_from_device(matrix_t Mhost, const matrix_t Mdevice)
{
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
    return;
}

/* Prints the matrix out to screen */
void print_matrix(const matrix_t M)
{
	for (unsigned int i = 0; i < M.num_rows; i++) {
        for (unsigned int j = 0; j < M.num_columns; j++) {
			printf("%f ", M.elements[i * M.num_columns + j]);
        }
		
        printf("\n");
	} 
	
    printf("\n");
    return;
}

/* Returns a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
    float r = rand()/(float)RAND_MAX;
	return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check for errors in kernel execution */
void check_CUDA_error(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if ( cudaSuccess != err) {
		printf("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}	
    
    return;    
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix(unsigned int num_rows, unsigned int num_columns)
{
	matrix_t M;
	M.num_columns = num_columns;
	M.num_rows = num_rows; 
	unsigned int size = M.num_rows * M.num_columns;
	M.elements = (float *)malloc(size * sizeof(float));
    if (M.elements == NULL)
        return M;

	/* Create a matrix with random numbers between [-.5 and .5] */
    unsigned int i, j;
	for (i = 0; i < size; i++)
        M.elements[i] = get_random_number (MIN_NUMBER, MAX_NUMBER);
	
	/* Make diagonal entries large with respect to the entries on each row. */
	for (i = 0; i < num_rows; i++) {
		float row_sum = 0.0;		
		for (j = 0; j < num_columns; j++) {
			row_sum += fabs(M.elements[i * M.num_rows + j]);
		}
		
        M.elements[i * M.num_rows + i] = 0.5 + row_sum;
	}

    return M;
}

