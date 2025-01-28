#!/bin/bash

threads="4 8 16"
sizes="10000 1000000 100000000"

rm -f saxpy
gcc -o saxpy saxpy.c -O3 -Wall -std=c99 -pthread -lm
if [ ! -f saxpy ]; then
    echo "Failed to compile saxpy program"
    exit 1
fi

for t in $threads; do
    echo "Threads: $t"
    for s in $sizes; do
        echo "Size: $s"
        ./saxpy $s $t | grep "time" | sed "s/s//g"
    done
    echo ""
done
