/* Reference code implementing the box blur filter.

    Build and execute as follows: 
        make clean && make 
        ./blur_filter size

    Author: Naga Kandasamy
    Date modified: February 20, 2025

    Student name(s): Cole Bardin
    Date modified: 2/27/25
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

/* #define DEBUG */

/* Include the kernel code */
#include "blur_filter_kernel.cu"

#define TILE_SIZE 32

extern "C" void compute_gold(const image_t, image_t);
void compute_on_device(const image_t, image_t);
int check_results(const float *, const float *, int, float);
void print_image(const image_t);

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s size\n", argv[0]);
        fprintf(stderr, "size: Height of the image. The program assumes size x size image.\n");
        exit(EXIT_FAILURE);
    }

    /* Allocate memory for the input and output images */
    int size = atoi(argv[1]);

    fprintf(stdout, "Creating %d x %d images\n", size, size);
    image_t in, out_gold, out_gpu;
    in.size = out_gold.size = out_gpu.size = size;
    in.element = (float *)malloc(sizeof(float) * size * size);
    out_gold.element = (float *)malloc(sizeof(float) * size * size);
    out_gpu.element = (float *)malloc(sizeof(float) * size * size);
    if ((in.element == NULL) || (out_gold.element == NULL) || (out_gpu.element == NULL)) {
        perror("Malloc");
        exit(EXIT_FAILURE);
    }

	struct timeval start, stop;

    /* Poplulate our image with random values between [-0.5 +0.5] */
    srand(time(NULL));
    int i;
    for (i = 0; i < size * size; i++)
        in.element[i] = rand()/(float)RAND_MAX -  0.5;
  
   /* Calculate the blur on the CPU. The result is stored in out_gold. */
    fprintf(stdout, "Calculating blur on the CPU\n"); 
	gettimeofday(&start, NULL);
    compute_gold(in, out_gold); 
	gettimeofday(&stop, NULL);
	printf("Execution time CPU = %f s\n", (float)(stop.tv_sec - start.tv_sec\
				+ (stop.tv_usec - start.tv_usec)/(float)1000000));

#ifdef DEBUG 
   print_image(in);
   print_image(out_gold);
#endif

   /* Calculate the blur on the GPU. The result is stored in out_gpu. */
   fprintf(stdout, "Calculating blur on the GPU\n");
	gettimeofday(&start, NULL);
   compute_on_device(in, out_gpu);
	gettimeofday(&stop, NULL);
	printf("Execution time GPU = %f s\n", (float)(stop.tv_sec - start.tv_sec\
				+ (stop.tv_usec - start.tv_usec)/(float)1000000));
#ifdef DEBUG
   print_image(in);
   print_image(out_gpu);
#endif

   /* Check CPU and GPU results for correctness */
   fprintf(stdout, "Checking CPU and GPU results\n");
   int num_elements = out_gold.size * out_gold.size;
   float eps = 1e-6;    /* Do not change */
   int check;
   check = check_results(out_gold.element, out_gpu.element, num_elements, eps);
   if (check == 0) 
       fprintf(stdout, "TEST PASSED\n");
   else
       fprintf(stdout, "TEST FAILED\n");
   
   /* Free data structures on the host */
   free((void *)in.element);
   free((void *)out_gold.element);
   free((void *)out_gpu.element);

    exit(EXIT_SUCCESS);
}

/* Calculate the blur on the GPU */
void compute_on_device(const image_t in, image_t out)
{
    // Allocate GPU memory
    float *igp;
    float *ogp;
    int size = in.size * in.size * sizeof(float);

    cudaMalloc((void**)&igp, size);
    if(igp == NULL){
        fprintf(stderr, "CUDA Malloc failed to allocate memory for input array\n");
        return;
    }
    cudaMalloc((void**)&ogp, size);
    if(ogp == NULL){
        cudaFree(igp);
        fprintf(stderr, "CUDA Malloc failed to allocate memory for output array\n");
        return;
    }
    cudaMemcpy(igp, in.element, size, cudaMemcpyHostToDevice);

    // Define thread blocks and shiz
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid((in.size + TILE_SIZE - 1)/TILE_SIZE, (in.size + TILE_SIZE - 1)/TILE_SIZE);

    // Call kernel
    blur_filter_kernel<<< grid, threads >>>(igp, ogp, in.size);
    cudaDeviceSynchronize();

    cudaMemcpy(out.element, ogp, size, cudaMemcpyDeviceToHost);

    cudaFree(igp);
    cudaFree(ogp);
    return;
}

/* Check correctness of results */
int check_results(const float *pix1, const float *pix2, int num_elements, float eps) 
{
    int i;
    for (i = 0; i < num_elements; i++)
        if (fabsf((pix1[i] - pix2[i])/pix1[i]) > eps) 
            return -1;
    
    return 0;
}

/* Print out the image contents */
void print_image(const image_t img)
{
    int i, j;
    float val;
    for (i = 0; i < img.size; i++) {
        for (j = 0; j < img.size; j++) {
            val = img.element[i * img.size + j];
            printf("%0.4f ", val);
        }
        printf("\n");
    }

    printf("\n");
}
