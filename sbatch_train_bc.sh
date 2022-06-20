#!/bin/bash
#SBATCH --mem-per-gpu=24G
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dineshj-high
#SBATCH --cpus-per-gpu=4
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH -w node-3090-0
#SBATCH --job-name=res2
#SBATCH -o out/res2_t20.out

export MODEL_TYPE="transfer"
export N_BLOCKS=2
export NET_TYPE="residual"
export TIME_INTERVAL=20
export LR=0.0004
export BETA=1
export EVAL_FREQ=2
export SAVE_FREQ=2
export VIS_FREQ=4000
export EPOCHS=200
export BATCH_SIZE=64
export VIS_SAMPLE_SIZE=5
export TASK_VIS_SAMPLE_SIZE=2
export NUM_WORKERS=4
export DATA_HOME_DIR='/scratch/junyao/Datasets/something_something_processed'
export DATETIME=06191830
export SAVE="cluster_model=${MODEL_TYPE}_blocks=${N_BLOCKS}_net=${NET_TYPE}_time=${TIME_INTERVAL}_lr=${LR}\
_beta=${BETA}_batch=${BATCH_SIZE}_date=${DATETIME}"
# export SAVE="cluster_test"

echo "MODEL_TYPE: ${MODEL_TYPE}"
echo "N_BLOCKS: ${N_BLOCKS}"
echo "NET_TYPE: ${NET_TYPE}"
echo "TIME_INTERVAL: ${TIME_INTERVAL}"
echo "LR: ${LR}"
echo "BETA: ${BETA}"
echo "EVAL_FREQ: ${EVAL_FREQ}"
echo "SAVE_FREQ: ${SAVE_FREQ}"
echo "VIS_FREQ: ${VIS_FREQ}"
echo "EPOCHS: ${EPOCHS}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "VIS_SAMPLE_SIZE: ${VIS_SAMPLE_SIZE}"
echo "TASK_VIS_SAMPLE_SIZE: ${TASK_VIS_SAMPLE_SIZE}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "DATA_HOME_DIR: ${DATA_HOME_DIR}"
echo "DATETIME: ${DATETIME}"
echo "SAVE: ${SAVE}"

xvfb-run -a python /home/junyao/LfHV/r3m/train_bc.py \
--model_type=${MODEL_TYPE} \
--n_blocks=${N_BLOCKS} \
--net_type=${NET_TYPE} \
--time_interval=${TIME_INTERVAL} \
--lr=${LR} \
--beta=${BETA} \
--eval_freq=${EVAL_FREQ} \
--save_freq=${SAVE_FREQ} \
--vis_freq=${VIS_FREQ} \
--epochs=${EPOCHS} \
--batch_size=${BATCH_SIZE} \
--vis_sample_size=${VIS_SAMPLE_SIZE} \
--task_vis_sample_size=${TASK_VIS_SAMPLE_SIZE} \
--num_workers=${NUM_WORKERS} \
--save=${SAVE} \
--data_home_dir=${DATA_HOME_DIR} \
--use_visualizer

wait