export MODEL_TYPE="resnet4"
export TIME_INTERVAL=20
export LR=0.001
export BETA=1
export EVAL_FREQ=1
export SAVE_FREQ=50
export EPOCHS=300
export BATCH_SIZE=5
export VIS_SAMPLE_SIZE=5
export TASK_VIS_SAMPLE_SIZE=0
export NUM_WORKERS=2
export DATETIME=06182245
export SAVE="cv_model=${MODEL_TYPE}_time=${TIME_INTERVAL}_lr=${LR}_beta=${BETA}_date=${DATETIME}_sanity_check"
# export SAVE="debug"

echo "MODEL_TYPE: ${MODEL_TYPE}"
echo "TIME_INTERVAL: ${TIME_INTERVAL}"
echo "LR: ${LR}"
echo "BETA: ${BETA}"
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
--time_interval=${TIME_INTERVAL} \
--lr=${LR} \
--beta=${BETA} \
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
