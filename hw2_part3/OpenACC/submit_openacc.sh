#!/bin/bash
#SBATCH -J tile
#SBATCH -e tile.err
#SBATCH -o tile.out
#SBATCH -p serial_requeue
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-00:20
#SBATCH --mem=64000
#SBATCH --job-name="tile"
#SBATCH --ntasks-per-node 2
#SBATCH --gres=gpu:1

pgcc -acc -o tile_OpenACC tile_OpenACC.c -Minfo=accel -ta=nvidia
./tile_OpenACC
