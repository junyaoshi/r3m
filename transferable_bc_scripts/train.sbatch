#!/bin/bash
#SBATCH --mem-per-gpu=24G
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dineshj-high
#SBATCH --cpus-per-gpu=4
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH -w node-3090-0
#SBATCH --job-name=b0.05
#SBATCH -o /home/junyao/transferable_out/t=15_nblocks=8_lambdas=[1,1,1,20]_bce=005.out

export N_BLOCKS=8; echo "N_BLOCKS: ${N_BLOCKS}"
export LAMBDA1=1; echo "LAMBDA1: ${LAMBDA1}"
export LAMBDA2=1; echo "LAMBDA2: ${LAMBDA2}"
export LAMBDA3=1; echo "LAMBDA3: ${LAMBDA3}"
export LAMBDA4=20; echo "LAMBDA4: ${LAMBDA4}"
export BCE_WEIGHT_MULT=0.05; echo "BCE_WEIGHT_MULT: ${BCE_WEIGHT_MULT}"
export TIME_INTERVAL=15; echo "TIME_INTERVAL: ${TIME_INTERVAL}"
export NET_TYPE="residual"; echo "NET_TYPE: ${NET_TYPE}"
export LR=0.0004; echo "LR: ${LR}"
export PRED_NODE="residual"; echo "PRED_NODE: ${PRED_NODE}"
export EVAL_FREQ=2; echo "EVAL_FREQ: ${EVAL_FREQ}"
export LOG_SCALAR_FREQ=100; echo "LOG_SCALAR_FREQ: ${LOG_SCALAR_FREQ}"
export SAVE_FREQ=10; echo "SAVE_FREQ: ${SAVE_FREQ}"
export VIS_FREQ=2000; echo "VIS_FREQ: ${VIS_FREQ}"
export EPOCHS=500; echo "EPOCHS: ${EPOCHS}"
export BATCH_SIZE=128; echo "BATCH_SIZE: ${BATCH_SIZE}"
export VIS_SAMPLE_SIZE=8; echo "VIS_SAMPLE_SIZE: ${VIS_SAMPLE_SIZE}"
export TASK_VIS_SAMPLE_SIZE=5; echo "TASK_VIS_SAMPLE_SIZE: ${TASK_VIS_SAMPLE_SIZE}"
export NUM_WORKERS=4; echo "NUM_WORKERS: ${NUM_WORKERS}"
export DATETIME=11082030; echo "DATETIME: ${DATETIME}"
export SAVE="${DATETIME}/cluster_t=${TIME_INTERVAL}_net=${NET_TYPE}_nblocks=${N_BLOCKS}_pred=${PRED_NODE}_lr=${LR}\
_lambdas=[${LAMBDA1},${LAMBDA2},${LAMBDA3},${LAMBDA4}]_bce=${BCE_WEIGHT_MULT}_batch=${BATCH_SIZE}"
#export SAVE="${DATETIME}/eval_on_train"
echo "SAVE: ${SAVE}"
export DATA_HOME_DIR="/scratch/junyao/Datasets/something_something_processed"; echo "DATA_HOME_DIR: ${DATA_HOME_DIR}"

xvfb-run -a python /home/junyao/LfHV/r3m/train_transferable_bc.py \
--n_blocks=${N_BLOCKS} \
--net_type=${NET_TYPE} \
--time_interval=${TIME_INTERVAL} \
--lr=${LR} \
--lambda1=${LAMBDA1} \
--lambda2=${LAMBDA2} \
--lambda3=${LAMBDA3} \
--lambda4=${LAMBDA4} \
--bce_weight_mult=${BCE_WEIGHT_MULT} \
--pred_mode=${PRED_NODE} \
--eval_freq=${EVAL_FREQ} \
--log_scalar_freq=${LOG_SCALAR_FREQ} \
--save_freq=${SAVE_FREQ} \
--vis_freq=${VIS_FREQ} \
--epochs=${EPOCHS} \
--eval_tasks \
--batch_size=${BATCH_SIZE} \
--vis_sample_size=${VIS_SAMPLE_SIZE} \
--task_vis_sample_size=${TASK_VIS_SAMPLE_SIZE} \
--num_workers=${NUM_WORKERS} \
--use_visualizer \
--save=${SAVE} \
--data_home_dir=${DATA_HOME_DIR} \

wait