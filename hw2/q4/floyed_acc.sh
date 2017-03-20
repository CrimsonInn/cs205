#!/bin/bash
#SBATCH -J floyed
#SBATCH -e floyed.err
#SBATCH -o floyed.out
#SBATCH -p serial_requeue
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-00:10
#SBATCH --mem=64000
#SBATCH --job-name="floyed"
#SBATCH --ntasks-per-node 2
#SBATCH --gres=gpu:1

pgcc -acc -o floyed floyed.c -Minfo=accel -ta=nvidia
./floyed
