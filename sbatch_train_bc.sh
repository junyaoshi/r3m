#!/bin/bash
#SBATCH --mem-per-gpu=24G
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dineshj-high
#SBATCH --cpus-per-gpu=4
#SBATCH --time=6:00:00
#SBATCH --gpus=1
#SBATCH -w node-3090-0
#SBATCH --job-name=r3m_bc
#SBATCH -o out/r3m_bc.out

export MODEL_TYPE="r3m_bc"
export LR=0.0004
export BATCH_SIZE=128
export SAVE="cluster_model=${MODEL_TYPE}_lr=${LR}_batch=${BATCH_SIZE}"
export EPOCHS=300
export NUM_WORKERS=2
export VIS_FREQ=3000
export SAVE_FREQ=2
export DATA_HOME_DIR='/scratch/junyao/Datasets/something_something_processed'

echo "MODEL_TYPE: ${MODEL_TYPE}"
echo "LR: ${LR}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "EPOCHS: ${EPOCHS}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "VIS_FREQ: ${VIS_FREQ}"
echo "SAVE: ${SAVE}"
echo "SAVE_FREQ: ${SAVE_FREQ}"
echo "DATA_HOME_DIR: ${DATA_HOME_DIR}"

xvfb-run -a python /home/junyao/LfHV/r3m/train_bc.py \
--model_type=${MODEL_TYPE} \
--lr=${LR} \
--batch_size=${BATCH_SIZE} \
--epochs=${EPOCHS} \
--num_workers=${NUM_WORKERS} \
--vis_freq=${VIS_FREQ} \
--save_freq=${SAVE_FREQ} \
--save=${SAVE} \
--data_home_dir=${DATA_HOME_DIR} \
--use_visualizer

wait