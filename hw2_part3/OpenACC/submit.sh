#!/bin/bash
#SBATCH -J qin
#SBATCH -e qin.err
#SBATCH -o qin.out
#SBATCH -p unrestricted
#SBATCH -c 64
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-00:10
#SBATCH --mem=64000
#SBATCH --job-name="matMatMultiV"

cc -O3 -fopenmp matrixMultiMatrix.c -o mmm.out -lm -ftree-vectorize -std=c99

thread_list='32 64'

for thread in $thread_list
do
    echo ============================
    echo testing with $thread threads
    export OMP_NUM_THREADS=$thread
    srun -c 64 ./mmm.out
done
