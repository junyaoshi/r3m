export MODEL_TYPE="r3m_bc"
export LR=0.0004
export SAVE="cv_model=${MODEL_TYPE}_lr=${LR}_sanity_check_fixed_vis"
export BATCH_SIZE=5
export VIS_SAMPLE_SIZE=2
export EPOCHS=300
export NUM_WORKERS=8
export SAVE_FREQ=100
export EVAL_FREQ=5

echo "MODEL_TYPE: ${MODEL_TYPE}"
echo "LR: ${LR}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "EPOCHS: ${EPOCHS}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "EVAL_FREQ: ${EVAL_FREQ}"
echo "SAVE: ${SAVE}"
echo "SAVE_FREQ: ${SAVE_FREQ}"
echo "VIS_SAMPLE_SIZE: ${VIS_SAMPLE_SIZE}"

CUDA_VISIBLE_DEVICES=1 xvfb-run -a python /home/junyao/LfHV/r3m/train_bc.py \
--model_type=${MODEL_TYPE} \
--lr=${LR} \
--batch_size=${BATCH_SIZE} \
--epochs=${EPOCHS} \
--num_workers=${NUM_WORKERS} \
--vis_sample_size=${VIS_SAMPLE_SIZE} \
--eval_freq=${EVAL_FREQ} \
--save_freq=${SAVE_FREQ} \
--save=${SAVE} \
--run_on_cv_server \
--use_visualizer \
--sanity_check \
