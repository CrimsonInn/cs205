#!/bin/bash
gcc-6 -O0 -fopenmp matrixMultiVec.c -o mmv.out -lm

thread_list='2 4'

for thread in $thread_list
do
    echo ============================
    echo testing with $thread threads
    export OMP_NUM_THREADS=$thread
    ./mmv.out 
done