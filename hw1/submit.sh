#!/bin/bash
#SBATCH -J qin
#SBATCH -e qin.err
#SBATCH -o qin.out
#SBATCH -p serial_requeue
#SBATCH -c 4
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-00:10
#SBATCH --mem-per-cpu=24000

module load intel/15.0.0-fasrc01
module load intel-mkl/11.0.0.079-fasrc02
icc -mkl blasMatMulti.c -o dgemm
./dgemm
