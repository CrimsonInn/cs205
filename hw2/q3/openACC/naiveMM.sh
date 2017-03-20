#!/bin/bash
#SBATCH -J naiveMM
#SBATCH -e naiveMM.err
#SBATCH -o naiveMM.out
#SBATCH -p serial_requeue
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-00:20
#SBATCH --mem=64000
#SBATCH --job-name="naiveMM"
#SBATCH --ntasks-per-node 2
#SBATCH --gres=gpu:1
pgcc -acc -o naiveMM naiveMM.c -Minfo=accel -ta=nvidia
./naiveMM
