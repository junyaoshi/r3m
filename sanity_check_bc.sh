export MODEL_TYPE="resnet16"
export TIME_INTERVAL=10
export LR=0.001
export EVAL_FREQ=20
export SAVE_FREQ=100
export EPOCHS=500
export BATCH_SIZE=5
export VIS_SAMPLE_SIZE=2
export NUM_WORKERS=8
export DATETIME=06131130
export SAVE="cv_model=${MODEL_TYPE}_time=${TIME_INTERVAL}_lr=${LR}_date=${DATETIME}_sanity_check"

echo "MODEL_TYPE: ${MODEL_TYPE}"
echo "TIME_INTERVAL: ${TIME_INTERVAL}"
echo "LR: ${LR}"
echo "EVAL_FREQ: ${EVAL_FREQ}"
echo "SAVE_FREQ: ${SAVE_FREQ}"
echo "EPOCHS: ${EPOCHS}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "VIS_SAMPLE_SIZE: ${VIS_SAMPLE_SIZE}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "DATETIME: ${DATETIME}"
echo "SAVE: ${SAVE}"

CUDA_VISIBLE_DEVICES=0 xvfb-run -a python /home/junyao/LfHV/r3m/train_bc.py \
--model_type=${MODEL_TYPE} \
--time_interval=${TIME_INTERVAL} \
--lr=${LR} \
--eval_freq=${EVAL_FREQ} \
--save_freq=${SAVE_FREQ} \
--epochs=${EPOCHS} \
--batch_size=${BATCH_SIZE} \
--vis_sample_size=${VIS_SAMPLE_SIZE} \
--num_workers=${NUM_WORKERS} \
--save=${SAVE} \
--run_on_cv_server \
--use_visualizer \
--sanity_check \
