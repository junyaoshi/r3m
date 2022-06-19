export NUM_WORKERS=4
export N_EVAL_SAMPLES=20
export MODEL_TYPE='r3m_bc'
export CKPT=0004
export CHECKPOINT="/home/junyao/LfHV/r3m/checkpoints/cluster_model=${MODEL_TYPE}_time=20_lr=0.0004_batch=64_workers=4_date=06152345/checkpoint_${CKPT}.pt"
export SAVE="${MODEL_TYPE}/cv_model=${MODEL_TYPE}_time=20_date=06152345_ckpt=${CKPT}_data=valid"
export DATA_HOME_DIR="/scratch/junyao/Datasets/something_something_processed"

echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "MODEL_TYPE: ${MODEL_TYPE}"
echo "CKPT: ${CKPT}"
echo "N_EVAL_SAMPLES: ${N_EVAL_SAMPLES}"
echo "CHECKPOINT: ${CHECKPOINT}"
echo "SAVE: ${SAVE}"
echo "DATA_HOME_DIR: ${DATA_HOME_DIR}"

xvfb-run -a python /home/junyao/LfHV/r3m/eval_bc.py \
--num_workers=${NUM_WORKERS} \
--n_eval_samples=${N_EVAL_SAMPLES} \
--use_visualizer \
--checkpoint=${CHECKPOINT} \
--save=${SAVE} \
--data_home_dir=${DATA_HOME_DIR} \
# --eval_on_train
