# Simple makefile for building PSO on xunil
# Author: Naga Kandasamy, January 31, 2025

CC		:= /usr/bin/gcc
CCFLAGS := -std=c99 -Wall -O3 -pthread 
LDLIBS := -lm

all: pso

pso: pso.o pso_utils.o optimize_gold.o optimize_using_pthreads.o 
	$(CC) -o pso pso.o pso_utils.o optimize_gold.o optimize_using_pthreads.o $(LDLIBS) $(CCFLAGS)

pso.o: pso.c pso.h
	$(CC) -c pso.c $(CCFLAGS)

pso_utils.o: pso_utils.c pso.h
	$(CC) -c pso_utils.c $(CCFLAGS)

optimize_gold.o: optimize_gold.c pso.h
	$(CC) -c optimize_gold.c $(CCFLAGS)

optimize_using_pthreads.o: optimize_using_pthreads.c pso.h 
	$(CC) -c optimize_using_pthreads.c $(CCFLAGS)

clean: 
	rm pso *.o


