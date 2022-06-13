#!/bin/bash
#SBATCH --mem-per-gpu=24G
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dineshj-high
#SBATCH --cpus-per-gpu=4
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH -w node-3090-0
#SBATCH --job-name=r3m_10
#SBATCH -o out/r3m_bc_t10_w0.out

export MODEL_TYPE="r3m_bc"
export TIME_INTERVAL=10
export LR=0.0004
export EVAL_FREQ=2
export SAVE_FREQ=2
export VIS_FREQ=2000
export EPOCHS=300
export BATCH_SIZE=128
export VIS_SAMPLE_SIZE=5
export NUM_WORKERS=0
export DATA_HOME_DIR='/scratch/junyao/Datasets/something_something_processed'
export SAVE="cluster_model=${MODEL_TYPE}_time=${TIME_INTERVAL}_lr=${LR}_batch=${BATCH_SIZE}_workers=${NUM_WORKERS}"

echo "MODEL_TYPE: ${MODEL_TYPE}"
echo "TIME_INTERVAL: ${TIME_INTERVAL}"
echo "LR: ${LR}"
echo "EVAL_FREQ: ${EVAL_FREQ}"
echo "SAVE_FREQ: ${SAVE_FREQ}"
echo "VIS_FREQ: ${VIS_FREQ}"
echo "EPOCHS: ${EPOCHS}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "VIS_SAMPLE_SIZE: ${VIS_SAMPLE_SIZE}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "DATA_HOME_DIR: ${DATA_HOME_DIR}"
echo "SAVE: ${SAVE}"

xvfb-run -a python /home/junyao/LfHV/r3m/train_bc.py \
--model_type=${MODEL_TYPE} \
--time_interval=${TIME_INTERVAL} \
--lr=${LR} \
--eval_freq=${EVAL_FREQ} \
--save_freq=${SAVE_FREQ} \
--vis_freq=${VIS_FREQ} \
--epochs=${EPOCHS} \
--batch_size=${BATCH_SIZE} \
--vis_sample_size=${VIS_SAMPLE_SIZE} \
--num_workers=${NUM_WORKERS} \
--save=${SAVE} \
--data_home_dir=${DATA_HOME_DIR} \
--use_visualizer

wait