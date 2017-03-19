#!/bin/bash
#SBATCH -J lp2d
#SBATCH -e lp2d.err
#SBATCH -o lp2d.out
#SBATCH -p serial_requeue
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-00:10
#SBATCH --mem=64000
#SBATCH --job-name="lp2d"
#SBATCH --ntasks-per-node 2
#SBATCH --gres=gpu:1

pgcc -acc -o lp2d laplace2d.c -Minfo=accel -ta=nvidia
./lp2d
