export MODEL_TYPE="r3m_bc"
export LR=0.0004
export SAVE="cv_model=${MODEL_TYPE}_lr=${LR}"
export BATCH_SIZE=64
export EPOCHS=300
export NUM_WORKERS=8
export VIS_FREQ=1000

echo "MODEL_TYPE: ${MODEL_TYPE}"
echo "LR: ${LR}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "EPOCHS: ${EPOCHS}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "VIS_FREQ: ${VIS_FREQ}"
echo "SAVE: ${SAVE}"

CUDA_VISIBLE_DEVICES=1 xvfb-run -a python /home/junyao/LfHV/r3m/train_bc.py \
--model_type=${MODEL_TYPE} \
--lr=${LR} \
--batch_size=${BATCH_SIZE} \
--epochs=${EPOCHS} \
--num_workers=${NUM_WORKERS} \
--vis_freq=${VIS_FREQ} \
--save=${SAVE} \
--run_on_cv_server --use_visualizer