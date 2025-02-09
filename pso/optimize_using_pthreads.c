/* Implementation of PSO using pthreads.
 *
 * Author: Naga Kandasamy
 * Date: January 31, 2025
 *
 */
#define _DEFAULT_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include <stdbool.h>
#include <unistd.h>
#include "pso.h"

typedef struct 
{
    int tid;
    char *function;
    swarm_t *swarm;
    float xmax;
    float xmin;
    int max_iter;
    int start;
    int stop;
    int num_threads;
} thread_arg_t;

int global_iter; /* Gloabl iteration counter controlled by main for thread sync */
int thread_ready; /* Counts how many threads have interated their particles */
int global_g; /* Best index but a global scope, set by thread 0, used by everyone else */
pthread_mutex_t mutex; /* Mutex to lock access to the thread ready counter */

void *swam_thread(void *args)
{
    thread_arg_t *arg = args;
    int i, j, iter, g;
    float w, c1, c2;
    float r1, r2;
    float curr_fitness;
    particle_t *particle, *gbest;
    int max_iter = arg->max_iter;
    swarm_t *swarm = arg->swarm;
    float xmax = arg->xmax;
    float xmin = arg->xmin;
    float best_fitness;

    w = 0.79;
    c1 = 1.49;
    c2 = 1.49;
    iter = 0;

    unsigned int seed = time(NULL); // Seed the random number generator 

    while (iter < max_iter) 
    {
        for (i = arg->start; i < arg->stop; i++)
        {
            particle = &swarm->particle[i];
            gbest = &swarm->particle[particle->g];  /* Best performing particle from last iteration */ 
            for (j = 0; j < particle->dim; j++) {   /* Update this particle's state */
                r1 = (float)rand_r(&seed)/(float)RAND_MAX;
                r2 = (float)rand_r(&seed)/(float)RAND_MAX;
                /* Update particle velocity */
                particle->v[j] = w * particle->v[j]\
                                 + c1 * r1 * (particle->pbest[j] - particle->x[j])\
                                 + c2 * r2 * (gbest->x[j] - particle->x[j]);
                /* Clamp velocity */
                if ((particle->v[j] < -fabsf(xmax - xmin)) || (particle->v[j] > fabsf(xmax - xmin))) 
                    particle->v[j] = uniform(-fabsf(xmax - xmin), fabsf(xmax - xmin));

                /* Update particle position */
                particle->x[j] = particle->x[j] + particle->v[j];
                if (particle->x[j] > xmax)
                    particle->x[j] = xmax;
                if (particle->x[j] < xmin)
                    particle->x[j] = xmin;
            } /* State update */

            /* Evaluate current fitness */
            pso_eval_fitness(arg->function, particle, &curr_fitness);

            /* Update pbest */
            if (curr_fitness < particle->fitness) 
            {
                particle->fitness = curr_fitness;
                for (j = 0; j < particle->dim; j++) particle->pbest[j] = particle->x[j];
            }
        }

        // Evaluate fitness of particles this thread owns
        g = -1;
        best_fitness = INFINITY;
        for (i = arg->start; i < arg->stop; i++) {
            particle = &swarm->particle[i];
            if (particle->fitness < best_fitness) {
                best_fitness = particle->fitness;
                g = i;
            }
        }
        // Check if thread found new best, update gloabl g val
        pthread_mutex_lock(&mutex);
        if(global_g == -1) best_fitness = INFINITY;
        else best_fitness = swarm->particle[global_g].fitness;
        if(swarm->particle[g].fitness < best_fitness) global_g = g;
        thread_ready++;
        pthread_mutex_unlock(&mutex);

        while(iter == global_iter) usleep(1);
        for (i = arg->start; i < arg->stop; i++) {
            particle = &swarm->particle[i];
            particle->g = global_g;
        }
        
        iter++;
    }
    return NULL;
}

int pso_solve_thread(char *function, swarm_t *swarm, float xmax, float xmin, int max_iter, int num_threads)
{
    int i;
    thread_arg_t *args;
    pthread_t *threads;
    int swarm_size, ppt, extra;
    int g = -1;
    swarm_size = swarm->num_particles;
    ppt = swarm_size / num_threads;
    extra = swarm_size % num_threads;
    if(ppt == 0)
    {
        ppt = 1;
        num_threads = swarm_size;
        extra = 0;
    }
    
    args = malloc(sizeof(thread_arg_t) * num_threads);
    if(!args)
    {
        fprintf(stderr, "ERROR: Failed to allocate memory for pthread arguments\n");
        exit(EXIT_FAILURE);
    }
    threads = malloc(sizeof(pthread_t) * num_threads);
    if(!threads){
        fprintf(stderr, "ERROR: Failed to allocate memory for pthread handles\n");
        exit(EXIT_FAILURE);
    }

#ifdef VERBOSE_DEBUG
    printf("particles: %d\n", swarm_size);
    printf("ppt: %d\n", ppt);
    printf("extra: %d\n", extra);
    printf("threads: %d\n", num_threads);
#endif

    thread_ready = 0;
    pthread_mutex_init(&mutex, NULL);
    global_iter = 0;
    global_g = -1;
    for(i = 0; i < num_threads; i++)
    {
        args[i].tid = i;
        args[i].function = function;
        args[i].swarm = swarm;
        args[i].xmax = xmax;
        args[i].xmin = xmin;
        args[i].max_iter = max_iter;
        args[i].start = i * ppt + (i < extra ? i : extra);
        args[i].stop = args[i].start + ppt + (i < extra ? 1 : 0);
        args[i].num_threads = num_threads;
#ifdef VERBOSE_DEBUG
        printf("thread %d: from %d to %d (count %d)\n", i, args[i].start, args[i].stop, args[i].stop - args[i].start);
#endif
        if(pthread_create(&threads[i], NULL, swam_thread, &args[i]) != 0)
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
    g = global_g;

    free(args);
    free(threads);
    pthread_mutex_destroy(&mutex);
    return g;
}

int optimize_using_pthreads(char *function, int dim, int swarm_size, 
                            float xmin, float xmax, int num_iter, int num_threads)
{
    int g = -1;

    /* Initialize PSO */
    swarm_t *swarm;
    srand(time(NULL));
    swarm = pso_init(function, dim, swarm_size, xmin, xmax);
    if (swarm == NULL) {
        fprintf(stderr, "Unable to initialize PSO\n");
        exit(EXIT_FAILURE);
    }

#ifdef VERBOSE_DEBUG
    pso_print_swarm(swarm);
#endif

    /* Solve PSO with threads */
    g = pso_solve_thread(function, swarm, xmax, xmin, num_iter, num_threads);
    if (g >= 0) {
        fprintf(stderr, "Solution:\n");
        pso_print_particle(&swarm->particle[g]);
    }

    pso_free(swarm);
    return g;
}


