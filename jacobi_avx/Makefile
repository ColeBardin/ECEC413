# Simple makefile for building Jacobi Solver on xunil
# Author: Naga Kandasamy, January 31, 2025

CC		:= /usr/bin/gcc
CCFLAGS := -std=c99 -Wall -O3 -mavx -pthread 
LDLIBS := -lm

all: jacobi_solver

jacobi_solver: jacobi_solver.o compute_gold.o compute_using_avx.o compute_using_pthread_avx.o 
	$(CC) -o jacobi_solver jacobi_solver.o compute_gold.o compute_using_avx.o compute_using_pthread_avx.o $(LDLIBS) $(CCFLAGS)

jacobi_solver.o: jacobi_solver.c jacobi_solver.h
	$(CC) -c jacobi_solver.c $(CCFLAGS)

compute_gold.o: compute_gold.c jacobi_solver.h
	$(CC) -c compute_gold.c $(CCFLAGS)

compute_using_avx.o: compute_using_avx.c jacobi_solver.h 
	$(CC) -c compute_using_avx.c $(CCFLAGS)

compute_using_pthread_avx.o: compute_using_pthread_avx.c jacobi_solver.h 
	$(CC) -c compute_using_pthread_avx.c $(CCFLAGS)

clean: 
	rm jacobi_solver *.o


