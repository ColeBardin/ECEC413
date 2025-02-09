# Particle Swarm Optimizer With Pthreads

Name: Cole Bardin
Date 2/09/25

## Building

There is an included `Makefile`. To compile, run:

`$ make`

To clean intermediate object and executable files, run:

`$ make clean`

## Running

There are 7 required command line arguments:

Usage: `$ ./pso function-name dimension swarm-size xmin xmax max-iter num-threads`
function-name: name of function to optimize
dimension: dimensionality of search space
swarm-size: number of particles in swarm
xmin, xmax: lower and upper bounds on search domain
max-iter: number of iterations to run the optimizer
num-threads: number of threads to create

The program will run the PSO twice, one single-threaded version and another threaded optimized version. The solution and execution times will be printed after each iteration.
