#!/bin/bash
#SBATCH -J qin
#SBATCH -e qin.err
#SBATCH -o qin.out
#SBATCH -p general
#SBATCH -c 8
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-00:10
#SBATCH --mem=64000
#SBATCH --job-name="matMatMultiV"

module load gcc/6.2.0-fasrc02
gcc -O0 -fopenmp matrixMultiMatrix.c -o mmm.out -lm -ftree-vectorize

thread_list='2 4 8'

for thread in $thread_list
do
    echo ============================
    echo testing with $thread threads
    export OMP_NUM_THREADS=$thread
    srun -c 8 ./mmm.out
done
