#!/bin/bash
#SBATCH -J qin
#SBATCH -e qin.err
#SBATCH -o qin.out
#SBATCH -p general
#SBATCH -c 8
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-00:30
#SBATCH --mem=64000
#SBATCH --job-name="benchmark"

#module load intel/15.0.0-fasrc01
#module load intel-mkl/11.0.0.079-fasrc02
module load gcc/5.3.0-fasrc01 OpenBLAS/0.2.18-fasrc01
#icc -mkl blasMatMulti.c -o dgemm
cc blasMatMulti.c -o dgemm -lopenblas -fopenmp
srun -c 8 ./dgemm

