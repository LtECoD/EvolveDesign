#!/bin/bash

#SBATCH --job-name maskmsa
#SBATCH -p gpu
#SBATCH -N 3
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 4
#SBATCH --cpus-per-task 4
#SBATCH --gpu-bind=none
#SBATCH --output ./log/maskmsa.out
#SBATCH --error ./log/maskmsa.err
#SBATCH --exclusive

export PYTHONPATH=`pwd`/src:${PYTHONPATH}
export OMP_NUM_THREADS=8
export NPROC_PER_NODE=4
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo ${MASTER_ADDR}:${MASTER_PORT}
echo $SLURM_NODEID

eval "$(conda shell.bash hook)"
conda activate heuristic

srun torchrun \
    --nnodes ${SLURM_NNODES} \
    --nproc_per_node $NPROC_PER_NODE \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    scripts/train/AFGen_train.py \
    scripts/train/configs/sequence.yaml
