#!/bin/bash
#SBATCH -J tile
#SBATCH -e tile.err
#SBATCH -o tile.out
#SBATCH -p serial_requeue
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda-7.5
#SBATCH -t 0-00:10
#SBATCH --mem-per-cpu=8000

module load cuda/7.5-fasrc01
#export OMP_NUM_THREADS=12
#gcc -fopenmp test.c -o test -O3
#srun -c 12 --threads-per-core 4 ./test
#srun -c 12 ./test
nvcc -o tile tile.cu -lcublas
./tile
