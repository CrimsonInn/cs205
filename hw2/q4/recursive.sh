#!/bin/bash
#SBATCH -J recursive
#SBATCH -e recursive.err
#SBATCH -o recursive.out
#SBATCH -p serial_requeue
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-00:10
#SBATCH --mem=64000
#SBATCH --job-name="recursive"
#SBATCH --ntasks-per-node 2
#SBATCH --gres=gpu:1

pgcc -acc -o recursive_acc recursive_acc.c -Minfo=accel -ta=nvidia
./recursive_acc
