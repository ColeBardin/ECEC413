#!/bin/bash

threads="2 4 8 16 32"
sizes="512 1024 2048"

make -s clean
make -s

for s in $sizes; do
    echo Size: $s
    for t in $threads; do
        echo Threads: $t 
        ./jacobi_solver $s $t | grep -i "time"
    done
done
