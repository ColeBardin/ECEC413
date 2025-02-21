/* Optimize Jacobi using pthread and AVX instructions */

#define _XOPEN_SOURCE 600
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
    matrix_t A;
    matrix_t x;
    matrix_t new_x;
    matrix_t B;
    int max_iter;
    int start;
    int stop;
} arg_t;

void *thread_body(void *arg);

int global_iter;
int stop_threads;
float global_ssd;
pthread_mutex_t mutex; 
pthread_barrier_t s0;
pthread_barrier_t s1;

void compute_using_pthread_avx(const matrix_t A, matrix_t pthread_avx_solution_x, const matrix_t B, int max_iter, int num_threads)
{
    int i;
    int num_rows = A.num_rows;
    /* Assume that N % (num_threads * VECTOR_SIZE) = 0 */
    int chunk_size = num_rows / num_threads;

    pthread_mutex_init(&mutex, NULL);
    pthread_barrier_init(&s0, NULL, num_threads);
    pthread_barrier_init(&s1, NULL, num_threads);

    /* FIXME testing */
    stop_threads = 0;

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
        argz[i].A = A;
        argz[i].x = pthread_avx_solution_x;
        argz[i].new_x = new_x;
        argz[i].B = B;
        argz[i].max_iter = max_iter;
        argz[i].start = i * chunk_size;
        argz[i].stop = (i + 1) * chunk_size;
        if(pthread_create(&threads[i], NULL, thread_body, (void *)&argz[i]) != 0)
        {
            fprintf(stderr, "ERROR: Failed to create pthread %d\n", i);
            exit(EXIT_FAILURE);
        }
    }

    for(i = 0; i < num_threads; i++) pthread_join(threads[i], NULL);

#ifdef PRINT
    if (global_iter < max_iter)
        fprintf(stderr, "\nConvergence achieved after %d iterations\n", global_iter);
    else
        fprintf(stderr, "\nMaximum allowed iterations reached\n");
#endif

    free(argz);
    free(threads);
    free(new_x.elements);
}

void *thread_body(void *arg)
{
    int i, j;
    int iter = 0;
    arg_t *args = (arg_t *)arg;
    int tid = args->tid;
    matrix_t A = args->A;
    matrix_t x = args->x;
    matrix_t new_x = args->new_x;
    matrix_t B = args->B;
    int max_iter = args->max_iter;
    int start = args->start;
    int stop = args->stop;
    int num_cols = A.num_columns;

    double ssd, mse;
    /* Setup the ping-pong buffers */
    float *src = x.elements;
    float *dest = new_x.elements;
    float *temp;

#ifdef DEBUG
    printf("DEBUG: Thread %d working on rows [%d - %d)\n", tid, start, stop);
#endif

    while(!stop_threads)
    {
        for (i = start; i < stop; i++) { // iterate through rows
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

        for (i = start; i < stop; i += VECTOR_SIZE) {
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

        pthread_mutex_lock(&mutex);
        global_ssd += ssd;
        pthread_mutex_unlock(&mutex);
        pthread_barrier_wait(&s0);
        iter++;
        if(tid == 0)
        {
            mse = sqrt(global_ssd); /* Mean squared error. */
            global_ssd = 0;
            global_iter = iter;
            if ((mse <= THRESHOLD) || (iter == max_iter))
            {
                stop_threads = 1;
#ifdef DEBUG
                printf("DEBUG: Stopping threads. MSE = %f\n", mse);
#endif
            }
        }

        pthread_barrier_wait(&s1);

        /* Flip the ping-pong buffers */
        temp = src;
        src = dest;
        dest = temp;
    }

    return NULL;
}

