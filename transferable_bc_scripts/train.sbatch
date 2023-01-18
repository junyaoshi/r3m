#!/bin/bash
#SBATCH --mem-per-gpu=24G
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dineshj-high
#SBATCH --cpus-per-gpu=4
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH -w node-3090-0
#SBATCH --job-name=pre8
#SBATCH -o /home/junyao/transferable_out/stage=pre_prob=false_t=10_nblocks=8_lambdas=[1,1,1].out

export N_BLOCKS=8; echo "N_BLOCKS: ${N_BLOCKS}"
export STAGE="pre"; echo "STAGE: ${STAGE}"
export PRED_PROB=False; echo "PRED_PROB: ${PRED_PROB}"
export NET_TYPE="residual"; echo "NET_TYPE: ${NET_TYPE}"
export TIME_INTERVAL=10; echo "TIME_INTERVAL: ${TIME_INTERVAL}"
export LR=0.0004; echo "LR: ${LR}"
export LAMBDA1=1; echo "LAMBDA1: ${LAMBDA1}"
export LAMBDA2=1; echo "LAMBDA2: ${LAMBDA2}"
export LAMBDA3=1; echo "LAMBDA3: ${LAMBDA3}"
export LAMBDA4=1; echo "LAMBDA4: ${LAMBDA4}"
export BCE_WEIGHT_MULT=1.0; echo "BCE_WEIGHT_MULT: ${BCE_WEIGHT_MULT}"
export PRED_RESIDUAL=True; echo "PRED_RESIDUAL: ${PRED_RESIDUAL}"
export PRED_CONTACT=False; echo "PRED_CONTACT: ${PRED_CONTACT}"
export EVAL_FREQ=2; echo "EVAL_FREQ: ${EVAL_FREQ}"
export LOG_SCALAR_FREQ=100; echo "LOG_SCALAR_FREQ: ${LOG_SCALAR_FREQ}"
export SAVE_FREQ=10; echo "SAVE_FREQ: ${SAVE_FREQ}"
export VIS_FREQ=2000; echo "VIS_FREQ: ${VIS_FREQ}"
export EPOCHS=500; echo "EPOCHS: ${EPOCHS}"
export BATCH_SIZE=128; echo "BATCH_SIZE: ${BATCH_SIZE}"
export VIS_SAMPLE_SIZE=8; echo "VIS_SAMPLE_SIZE: ${VIS_SAMPLE_SIZE}"
export TASK_VIS_SAMPLE_SIZE=5; echo "TASK_VIS_SAMPLE_SIZE: ${TASK_VIS_SAMPLE_SIZE}"
export NUM_WORKERS=4; echo "NUM_WORKERS: ${NUM_WORKERS}"
export DATETIME=01180200; echo "DATETIME: ${DATETIME}"
export SAVE="${DATETIME}/cluster_t=${TIME_INTERVAL}_net=${NET_TYPE}_nblocks=${N_BLOCKS}_stage=${STAGE}\
_predres=${PRED_NODE}_contact=${PRED_CONTACT}_lr=${LR}_lambdas=[${LAMBDA1},${LAMBDA2},${LAMBDA3}]_batch=${BATCH_SIZE}"
#export SAVE="${DATETIME}/eval_on_train"
echo "SAVE: ${SAVE}"
export DATA_HOME_DIR="/scratch/junyao/Datasets/something_something_processed"; echo "DATA_HOME_DIR: ${DATA_HOME_DIR}"

xvfb-run -a python /home/junyao/LfHV/r3m/train_transferable_bc.py \
--n_blocks=${N_BLOCKS} \
--net_type=${NET_TYPE} \
--pred_residual=${PRED_RESIDUAL} \
--pred_contact=${PRED_CONTACT} \
--pred_prob=${PRED_PROB} \
--time_interval=${TIME_INTERVAL} \
--stage=${STAGE} \
--lr=${LR} \
--lambda1=${LAMBDA1} \
--lambda2=${LAMBDA2} \
--lambda3=${LAMBDA3} \
--lambda4=${LAMBDA4} \
--bce_weight_mult=${BCE_WEIGHT_MULT} \
--eval_freq=${EVAL_FREQ} \
--log_scalar_freq=${LOG_SCALAR_FREQ} \
--save_freq=${SAVE_FREQ} \
--vis_freq=${VIS_FREQ} \
--epochs=${EPOCHS} \
--eval_tasks=True \
--batch_size=${BATCH_SIZE} \
--vis_sample_size=${VIS_SAMPLE_SIZE} \
--task_vis_sample_size=${TASK_VIS_SAMPLE_SIZE} \
--num_workers=${NUM_WORKERS} \
--use_visualizer=True \
--save=${SAVE} \
--data_home_dir=${DATA_HOME_DIR} \

wait