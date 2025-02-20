/* Optimize Jacobi using pthread and AVX instructions */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <immintrin.h>
#include "jacobi_solver.h"

#define VECTOR_SIZE 8 /* AVX operates on 8 single-precision floating-point values */

void compute_using_pthread_avx(const matrix_t A, matrix_t pthread_avx_solution_x, const matrix_t B, int max_iter, int num_threads)
{
}



