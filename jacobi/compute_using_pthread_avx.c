/* Optimize Jacobi using pthread and AVX instructions */

#define _POSIX_SOURCE
#define _BSD_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <immintrin.h>
#include <unistd.h>
#include <time.h>
#include "jacobi_solver.h"

#define VECTOR_SIZE 8 /* AVX operates on 8 single-precision floating-point values */

typedef struct 
{
    int tid;
    matrix_t *A;
    matrix_t *x;
    matrix_t *new_x;
    matrix_t *B;
    int max_iter;
    int start;
    int stop;
} arg_t;

void *thread_body(void *arg);

int global_iter;
int thread_ready; 
int global_g; 
pthread_mutex_t mutex; 

void compute_using_pthread_avx(const matrix_t A, matrix_t pthread_avx_solution_x, const matrix_t B, int max_iter, int num_threads)
{
    int i;
    int num_rows = A.num_rows;
    long num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    /* Don't use more threads than cores since that defeats the purpose of parallelism */
    if(num_threads > num_cpus) num_threads = num_cpus;
    /* Assume that N % (num_threads * VECTOR_SIZE) = 0 */
    int chunk_size = num_rows / (num_threads * VECTOR_SIZE);

    /* FIXME testing */
    max_iter = 1;
    global_iter = 0;

    /* Allocate n x 1 matrix to hold iteration values.*/
    matrix_t new_x = allocate_matrix(num_rows, 1, 0);      

    /* Initialize current jacobi solution. */
    for (i = 0; i < num_rows; i++) pthread_avx_solution_x.elements[i] = B.elements[i];

    arg_t *argz = (arg_t *)malloc(sizeof(arg_t) * num_threads);
    if(!argz)
    {
        perror("Failed to allocate argument array");
        exit(EXIT_FAILURE);
    }
    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * num_threads);
    if(!threads){
        perror("Failed to allocate thread array");
        exit(EXIT_FAILURE);
    }
    for(i = 0; i < num_threads; i++)
    {
        argz[i].tid = i;
        argz[i].A = &A;
        argz[i].x = &pthread_avx_solution_x;
        argz[i].new_x = &new_x;
        argz[i].B = &B;
        argz[i].max_iter = max_iter;
        argz[i].start = i * chunk_size;
        argz[i].stop = (i + 1) * chunk_size;
        if(pthread_create(&threads[i], NULL, thread_body, (void *)&argz[i]) != 0)
        {
            fprintf(stderr, "ERROR: Failed to create pthread %d\n", i);
            exit(EXIT_FAILURE);
        }
    }

    while(global_iter < max_iter)
    {
        while(thread_ready < num_threads) usleep(1);
        thread_ready = 0;
        global_iter++;
    }

    for(i = 0; i < num_threads; i++) pthread_join(threads[i], NULL);

    free(argz);
    free(threads);
    free(new_x.elements);
}

void *thread_body(void *arg)
{
    int i;
    int iter = 0;
    arg_t *args = (arg_t *)arg;
    int tid = args->tid;
    matrix_t *A = args->A;
    matrix_t *x = args->x;
    matrix_t *new_x = args->new_x;
    matrix_t *B = args->B;
    int max_iter = args->max_iter;
    int start = args->start;
    int stop = args->stop;

    while(iter < max_iter)
    {
        for (i = start; i < stop; i++) {
            float sum = -A.elements[i * num_cols + i] * src[i];

            __m256 sum_vec = _mm256_setzero_ps(); // AVX sum accumulator

            for (j = 0; j < num_cols; j += VECTOR_SIZE) { // Process 8 elements at a time
                __m256 a_vec = _mm256_loadu_ps(&A.elements[i * num_cols + j]);
                __m256 x_vec = _mm256_loadu_ps(&src[j]);
                __m256 ax_vec = _mm256_mul_ps(a_vec, x_vec);
                sum_vec = _mm256_add_ps(sum_vec, ax_vec);
            }

            /* Horizontal sum of AVX register */
            float sum_array[8] __attribute__((aligned(32)));
            _mm256_storeu_ps(sum_array, sum_vec);
            sum += sum_array[0];
            sum += sum_array[1];
            sum += sum_array[2];
            sum += sum_array[3];
            sum += sum_array[4];
            sum += sum_array[5];
            sum += sum_array[6];
            sum += sum_array[7];

            /* Compute new value */
            dest[i] = (B.elements[i] - sum) / A.elements[i * num_cols + i];
        }

        /* Compute Mean Squared Error using AVX */
        __m256 ssd_vec = _mm256_setzero_ps();

        for (i = 0; i < num_rows; i += VECTOR_SIZE) {
            __m256 src_vec = _mm256_loadu_ps(&src[i]);
            __m256 dest_vec = _mm256_loadu_ps(&dest[i]);
            __m256 diff = _mm256_sub_ps(dest_vec, src_vec);
            __m256 square = _mm256_mul_ps(diff, diff); 
            ssd_vec = _mm256_add_ps(ssd_vec, square);
        }

        /* Reduce SSD */
        float ssd_array[8] __attribute__((aligned(32)));
        _mm256_storeu_ps(ssd_array, ssd_vec);
        ssd  = ssd_array[0];
        ssd += ssd_array[1];
        ssd += ssd_array[2];
        ssd += ssd_array[3];
        ssd += ssd_array[4];
        ssd += ssd_array[5];
        ssd += ssd_array[6];
        ssd += ssd_array[7];
        // barrier sync
        pthread_mutex_lock(&mutex);
        thread_ready++;
        pthread_mutex_unlock(&mutex);
        while(iter == global_iter) usleep(1);

        iter++;
    }

    return NULL;
}

