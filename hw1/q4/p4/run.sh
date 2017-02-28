#!/bin/bash

cc -O3 -fopenmp matrixMultiMatrix.c -o mmm.out -lm -ftree-vectorize -std=c99

thread_list='2 4 8'

for thread in $thread_list
do
    echo ============================
    echo testing with $thread threads
    export OMP_NUM_THREADS=$thread
    ./mmm.out 
done
