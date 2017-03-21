#!/bin/bash
#SBATCH -J bfs
#SBATCH -e bfs.err
#SBATCH -o bfs.out
#SBATCH -p serial_requeue
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-00:10
#SBATCH --mem=64000
#SBATCH --job-name="bfs"
#SBATCH --ntasks-per-node 2
#SBATCH --gres=gpu:1

pgcc -acc -o bfs bfs.c -Minfo=accel -ta=nvidia
./bfs
