#ifndef _JACOBI_ITERATION_H_
#define _JACOBI_ITERATION_H_

#define THRESHOLD 1e-5      /* Threshold for convergence */
#define MIN_NUMBER 2        /* Min number in the A and b matrices */
#define MAX_NUMBER 10       /* Max number in the A and b matrices */
#define MAX_ITER 1000000 

#define THREAD_BLOCK_SIZE 128           /* Size of a thread block */

#define MATRIX_SIZE 512
#define NUM_COLUMNS MATRIX_SIZE         /* Number of columns in matrix A */
#define NUM_ROWS MATRIX_SIZE            /* Number of rows in matrix A */

/* Matrix structure declaration */
typedef struct matrix_s {
    unsigned int num_columns;           /* Matrix width */
    unsigned int num_rows;              /* Matrix height */ 
    float *elements;
}  matrix_t;

/* Function prototypes */
extern "C" void compute_gold(const matrix_t, matrix_t, const matrix_t);
extern "C" void display_jacobi_solution(const matrix_t, const matrix_t, const matrix_t);
void solve_cuda_naive(const matrix_t A, matrix_t x, const matrix_t B);
void solve_cuda_optimized(const matrix_t A, matrix_t x, const matrix_t B);
matrix_t allocate_matrix_on_device(const matrix_t);
matrix_t allocate_matrix_on_host(int, int, int);
void matrix_transpose(const matrix_t, matrix_t);
int check_if_diagonal_dominant(const matrix_t);
matrix_t create_diagonally_dominant_matrix(unsigned int, unsigned int);
void copy_matrix_to_device(matrix_t, const matrix_t);
void copy_matrix_from_device(matrix_t, const matrix_t);
void compute_on_device(const matrix_t, matrix_t, matrix_t, const matrix_t);
int perform_simple_check(const matrix_t);
void print_matrix(const matrix_t);
float get_random_number(int, int);
void check_CUDA_error(const char *);

#endif /* _JACOBI_ITERATION_H_ */

