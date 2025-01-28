/* Implementation of the SAXPY loop.
 *
 * Compile as follows: gcc -o saxpy saxpy.c -O3 -Wall -std=c99 -pthread -lm
 *
 * Author: Naga Kandasamy
 * Date modified: January 23, 2025 
 *
 * Student names: Cole Bardin
 * Date: 1/31/25
 *
 * */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>

typedef struct 
{
    float *x;
    float *y;
    float a;
    int start;
    int chunk_size;
} chunk_args_t;

typedef struct
{
    float *x;
    float *y;
    float a;
    int tid;
    int k;
    int n;
} stride_args_t;

/* Function prototypes */
void compute_gold(float *, float *, float, int);
void compute_using_pthreads_v1(float *, float *, float, int, int);
void compute_using_pthreads_v2(float *, float *, float, int, int);
int check_results(float *, float *, int, float);
void *func_chunk(void *arg);
void *func_stride(void *arg);

int main(int argc, char **argv)
{
	if (argc < 3) {
		fprintf(stderr, "Usage: %s num-elements num-threads\n", argv[0]);
        fprintf(stderr, "num-elements: Number of elements in the input vectors\n");
        fprintf(stderr, "num-threads: Number of threads\n");
		exit(EXIT_FAILURE);
	}
	
    int num_elements = atoi(argv[1]); 
    int num_threads = atoi(argv[2]);

	/* Create vectors X and Y and fill them with random numbers between [-.5, .5] */
    printf("Generating input vectors\n");
    int i;
	float *x = (float *)malloc(sizeof(float) * num_elements);
    float *y1 = (float *)malloc(sizeof(float) * num_elements);              /* For the reference version */
	float *y2 = (float *)malloc(sizeof(float) * num_elements);              /* For pthreads version 1 */
	float *y3 = (float *)malloc(sizeof(float) * num_elements);              /* For pthreads version 2 */

	srand(time(NULL)); /* Seed random number generator */
	for (i = 0; i < num_elements; i++) {
		x[i] = rand()/(float)RAND_MAX - 0.5;
		y1[i] = rand()/(float)RAND_MAX - 0.5;
        y2[i] = y1[i]; /* Make copies of y1 for y2 and y3 */
        y3[i] = y1[i]; 
	}

    float a = 2.5;  /* Choose some scalar value for a */

	/* Calculate SAXPY using the reference solution. The resulting values are placed in y1 */
    printf("\nCalculating SAXPY using reference solution\n");
	struct timeval start, stop;	
	gettimeofday(&start, NULL);
	
    compute_gold(x, y1, a, num_elements); 
	
    gettimeofday(&stop, NULL);
	printf("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));

	/* Compute SAXPY using pthreads, version 1. Results must be placed in y2 */
    printf("\nCalculating SAXPY using pthreads, version 1\n");
	gettimeofday(&start, NULL);

	compute_using_pthreads_v1(x, y2, a, num_elements, num_threads);
	
    gettimeofday(&stop, NULL);
	printf("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));

    /* Compute SAXPY using pthreads, version 2. Results must be placed in y3 */
    printf("\nCalculating SAXPY using pthreads, version 2\n");
	gettimeofday(&start, NULL);

	compute_using_pthreads_v2(x, y3, a, num_elements, num_threads);
	
    gettimeofday(&stop, NULL);
	printf("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));

    /* Check results for correctness */
    printf("\nChecking results for correctness\n");
    float eps = 1e-12;                                      /* Do not change this value */
    if (check_results(y1, y2, num_elements, eps) == 0)
        printf("TEST PASSED\n");
    else 
        printf("TEST FAILED\n");
 
    if (check_results(y1, y3, num_elements, eps) == 0) printf("TEST PASSED\n");
    else printf("TEST FAILED\n");

	/* Free memory */ 
	free((void *)x);
	free((void *)y1);
    free((void *)y2);
	free((void *)y3);

    exit(EXIT_SUCCESS);
}

/* Compute reference soution using a single thread */
void compute_gold(float *x, float *y, float a, int num_elements)
{
	int i;
    for (i = 0; i < num_elements; i++)
        y[i] = a * x[i] + y[i]; 
}

/* Calculate SAXPY using pthreads, version 1. Place result in the Y vector */
void compute_using_pthreads_v1(float *x, float *y, float a, int num_elements, int num_threads)
{
    int i;
    int ret;
    int chunk_size;
    pthread_t *threads;
    chunk_args_t *args;
    int extra;
    bool need_extra;

    threads = (pthread_t *)malloc(sizeof(pthread_t) * num_threads);
    if(!threads)
    {
        fprintf(stderr, "ERROR: Failed to malloc array to store %d threads for chunk\n", num_threads);
        goto FAIL_C_MTHREAD;
    }
    args = (chunk_args_t *)malloc(sizeof(chunk_args_t) * num_threads);
    if(!args)
    {
        fprintf(stderr, "ERROR: Failed to malloc array to store %d thread args for chunk\n", num_threads);
        goto FAIL_C_MARG;
    }

    need_extra = false;
    chunk_size = num_elements / num_threads;
    if(num_elements % chunk_size != 0)
    {
        extra = num_elements - (chunk_size * num_threads);
        need_extra = true;
    }
    for(i = 0; i < num_threads; i++)
    {
        args[i].x = x;
        args[i].y = y;
        args[i].a = a;
        args[i].start = i * chunk_size;
        args[i].chunk_size = chunk_size + ((i == num_threads - 1) && need_extra ? extra : 0);
        ret = pthread_create(&threads[i], NULL, &func_chunk, &args[i]);
        if(ret != 0)
        {
            fprintf(stderr, "ERROR: Failed to create pthread #%d for chunk\n", i);
            goto FAIL_C_TCREATE;
        }
    }

    for(i = 0; i < num_threads; i++) pthread_join(threads[i], NULL);
    free(args); 
    free(threads);
    return;

FAIL_C_TCREATE:
    free(args);
FAIL_C_MARG:
    free(threads);
FAIL_C_MTHREAD:
    exit(EXIT_FAILURE);
}

/* Calculate SAXPY using pthreads, version 2. Place result in the Y vector */
void compute_using_pthreads_v2(float *x, float *y, float a, int num_elements, int num_threads)
{
    int i;
    int ret;
    pthread_t *threads;
    stride_args_t *args;

    threads = (pthread_t *)malloc(sizeof(pthread_t) * num_threads);
    if(!threads)
    {
        fprintf(stderr, "ERROR: Failed to malloc array to store %d threads for stride\n", num_threads);
        goto FAIL_S_MTHREAD;
    }
    args = (stride_args_t *)malloc(sizeof(stride_args_t) * num_threads);
    if(!args)
    {
        fprintf(stderr, "ERROR: Failed to malloc array to store %d thread args for stride\n", num_threads);
        goto FAIL_S_MARG;
    }

    for(i = 0; i < num_threads; i++)
    {
        args[i].x = x;
        args[i].y = y;
        args[i].a = a;
        args[i].tid = i;
        args[i].k = num_threads;
        args[i].n = num_elements;
        ret = pthread_create(&threads[i], NULL, &func_stride, &args[i]);
        if(ret != 0)
        {
            fprintf(stderr, "ERROR: Failed to create pthread #%d for stride\n", i);
            goto FAIL_S_TCREATE;
        }
    }

    for(i = 0; i < num_threads; i++) pthread_join(threads[i], NULL);
    free(args); 
    free(threads);
    return;

FAIL_S_TCREATE:
    free(args);
FAIL_S_MARG:
    free(threads);
FAIL_S_MTHREAD:
    exit(EXIT_FAILURE);
}

/* Perform element-by-element check of vector if relative error is within specified threshold */
int check_results(float *A, float *B, int num_elements, float threshold)
{
    int i;
    for (i = 0; i < num_elements; i++) {
        if (fabsf((A[i] - B[i])/A[i]) > threshold)
            return -1;
    }
    
    return 0;
}

void *func_chunk(void *arg)
{
    chunk_args_t *cargs = (chunk_args_t *)arg;
    float *x = cargs->x;
    float *y = cargs->y;
    float a = cargs->a;
    int start = cargs->start;
    int chunk_size = cargs->chunk_size;

    for(int i = 0; i < chunk_size; i++) y[start + i] = a * x[start + i] + y[start + i];
    return NULL;
}

void *func_stride(void *arg)
{
    stride_args_t *sargs = (stride_args_t *)arg;
    float *x = sargs->x;
    float *y = sargs->y;
    float a = sargs->a;
    int tid = sargs->tid;
    int k = sargs->k;
    int n = sargs->n;

    while(tid < n)
    {
        y[tid] = a * x[tid] + y[tid];
        tid = tid + k;
    }
    return NULL;
}

/*
typedef struct
{
    float *x;
    float *y;
    float a;
    int tid;
    int k;
    int n;
} stride_args_t;
*/

