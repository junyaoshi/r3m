export NUM_WORKERS=4
export N_EVAL_SAMPLES=29
export DATASET="same_hand_demos"
export CKPT_NUM=0150
export EVALTIME="07251430"
export DATA_HOME_DIR="/home/junyao/Datasets/something_something_hand_demos_same_hand"
export CHECKPOINT="/scratch/junyao/LfHV/r3m/checkpoints/transfer/cluster_model=transfer_blocks=4_net=residual_\
time=20_lr=0.0004_lambdas=[1,5,0]_batch=64_date=06261730/checkpoint_${CKPT_NUM}.pt"
export SAVE="cluster_model=transfer_blocks=4_net=residual_\
time=20_lr=0.0004_lambdas=[1,5,0]_batch=64_date=06261730_checkpoint_${CKPT_NUM}_evaldata=${DATASET}_evaltime=${EVALTIME}"

echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "N_EVAL_SAMPLES: ${N_EVAL_SAMPLES}"
echo "DATASET: ${DATASET}"
echo "CKPT_NUM: ${CKPT_NUM}"
echo "EVALTIME: ${EVALTIME}"
echo "DATA_HOME_DIR: ${DATA_HOME_DIR}"
echo "CHECKPOINT: ${CHECKPOINT}"
echo "SAVE: ${SAVE}"

CUDA_VISIBLE_DEVICES=1 xvfb-run -a python /home/junyao/LfHV/r3m/eval_bc.py \
--num_workers=${NUM_WORKERS} \
--n_eval_samples=${N_EVAL_SAMPLES} \
--checkpoint=${CHECKPOINT} \
--save=${SAVE} \
--data_home_dir=${DATA_HOME_DIR} \
--run_on_cv_server \
--dataset=${DATASET} \
--use_visualizer \
--eval_r3m \
--eval_task \
--log_depth \
--log_depth_scatter_plots \
--use_current_frame_info \
--no_shuffle