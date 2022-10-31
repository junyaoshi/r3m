export MODEL_TYPE="transfer"
export N_BLOCKS=4
export NET_TYPE="residual"
export TIME_INTERVAL=20
export LR=0.001
export LAMBDA1=0
export LAMBDA2=0
export LAMBDA3=1
export EVAL_FREQ=5
export SAVE_FREQ=50
export EPOCHS=300
export BATCH_SIZE=5
export VIS_SAMPLE_SIZE=5
export TASK_VIS_SAMPLE_SIZE=0
export NUM_WORKERS=2
export DATETIME=06261645
export SAVE="sanity_check/cv_model=${MODEL_TYPE}_blocks=${N_BLOCKS}_net=${NET_TYPE}_time=${TIME_INTERVAL}_lr=${LR}\
_lambdas=[${LAMBDA1},${LAMBDA2},${LAMBDA3}]_date=${DATETIME}_sanity_check"
# export SAVE="debug"

echo "MODEL_TYPE: ${MODEL_TYPE}"
echo "N_BLOCKS: ${N_BLOCKS}"
echo "NET_TYPE: ${NET_TYPE}"
echo "TIME_INTERVAL: ${TIME_INTERVAL}"
echo "LR: ${LR}"
echo "LAMBDA1: ${LAMBDA1}"
echo "LAMBDA2: ${LAMBDA2}"
echo "LAMBDA3: ${LAMBDA3}"
echo "EVAL_FREQ: ${EVAL_FREQ}"
echo "SAVE_FREQ: ${SAVE_FREQ}"
echo "EPOCHS: ${EPOCHS}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "VIS_SAMPLE_SIZE: ${VIS_SAMPLE_SIZE}"
echo "TASK_VIS_SAMPLE_SIZE: ${TASK_VIS_SAMPLE_SIZE}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "DATETIME: ${DATETIME}"
echo "SAVE: ${SAVE}"

CUDA_VISIBLE_DEVICES=1 xvfb-run -a python /home/junyao/LfHV/r3m/train_bc.py \
--model_type=${MODEL_TYPE} \
--n_blocks=${N_BLOCKS} \
--net_type=${NET_TYPE} \
--time_interval=${TIME_INTERVAL} \
--lr=${LR} \
--lambda1=${LAMBDA1} \
--lambda2=${LAMBDA2} \
--lambda3=${LAMBDA3} \
--eval_freq=${EVAL_FREQ} \
--save_freq=${SAVE_FREQ} \
--epochs=${EPOCHS} \
--batch_size=${BATCH_SIZE} \
--vis_sample_size=${VIS_SAMPLE_SIZE} \
--task_vis_sample_size=${TASK_VIS_SAMPLE_SIZE} \
--num_workers=${NUM_WORKERS} \
--save=${SAVE} \
--run_on_cv_server \
--use_visualizer \
--sanity_check \
