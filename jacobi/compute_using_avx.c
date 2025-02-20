/* Solve Jacobi using AVX instructions */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include "jacobi_solver.h"

/* FIXME: Complete this function to perform the Jacobi calculation using AVX. 
 * Result must be placed in avx_solution_x. */
void compute_using_avx(const matrix_t A, matrix_t avx_solution_x, const matrix_t B, int max_iter)
{
    int i, j;
    int num_rows = A.num_rows;
    int num_cols = A.num_columns;

    /* Allocate n x 1 matrix to hold iteration values.*/
    matrix_t new_x = allocate_matrix(num_rows, 1, 0);      

    /* Initialize current jacobi solution. */
    for (i = 0; i < num_rows; i++) avx_solution_x.elements[i] = B.elements[i];

    /* Setup the ping-pong buffers */
    float *src = avx_solution_x.elements;
    float *dest = new_x.elements;
    float *temp;

    /* Perform Jacobi iteration. */
    int done = 0;
    double ssd, mse;
    int num_iter = 0;

    while (!done) {
        for (i = 0; i < num_rows; i++) {
            float sum = -A.elements[i * num_cols + i] * src[i];

            __m128 sum_vec = _mm_setzero_ps(); // AVX sum accumulator

            for (j = 0; j < num_cols; j += 4) { // Process 4 elements at a time
                __m128 a_vec = _mm_loadu_ps(&A.elements[i * num_cols + j]);
                __m128 x_vec = _mm_loadu_ps(&src[j]);
                __m128 ax_vec = _mm_mul_ps(a_vec, x_vec);
                sum_vec = _mm_add_ps(sum_vec, ax_vec);
            }

            /* Horizontal sum of AVX register */
            float sum_array[4] __attribute__((aligned(16)));
            _mm_store_ps(sum_array, sum_vec);
            sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

            /* Compute new value */
            dest[i] = (B.elements[i] - sum) / A.elements[i * num_cols + i];
        }

        /* Compute Mean Squared Error using AVX */
        __m128 ssd_vec = _mm_setzero_ps();

        for (i = 0; i < num_rows; i += 4) {
            __m128 src_vec = _mm_loadu_ps(&src[i]);
            __m128 dest_vec = _mm_loadu_ps(&dest[i]);
            __m128 diff = _mm_sub_ps(dest_vec, src_vec);
            __m128 square = _mm_mul_ps(diff, diff); 
            ssd_vec = _mm_add_ps(ssd_vec, square);
        }

        /* Reduce SSD */
        float ssd_array[4] __attribute__((aligned(16)));
        _mm_store_ps(ssd_array, ssd_vec);
        ssd = ssd_array[0] + ssd_array[1] + ssd_array[2] + ssd_array[3];

        num_iter++;
        mse = sqrt(ssd); /* Mean squared error. */
#ifdef PRINT
        fprintf(stderr, "Iteration: %d. MSE = %f\n", num_iter, mse);
#endif

        if ((mse <= THRESHOLD) || (num_iter == max_iter))
            done = 1;

        /* Flip the ping-pong buffers */
        temp = src;
        src = dest;
        dest = temp;
    }

#ifdef PRINT
    if (num_iter < max_iter)
        fprintf(stderr, "\nConvergence achieved after %d iterations\n", num_iter);
    else
        fprintf(stderr, "\nMaximum allowed iterations reached\n");
#endif

    free(new_x.elements);
}


