export NUM_WORKERS=4
export N_EVAL_SAMPLES=1000
export DATASET="robot_demos"
export IOU_THRESH=0.6
export CKPT_NUM=0150
export EVALTIME="09201530"
export DATA_HOME_DIR="/home/junyao/Datasets/something_something_hand_robot_paired/widowx"
export CHECKPOINT="/home/junyao/LfHV/r3m/checkpoints/transfer/cluster_model=transfer_blocks=4_net=residual_\
time=20_lr=0.0004_lambdas=[1,5,0]_batch=64_date=06261730/checkpoint_${CKPT_NUM}.pt"
export ROOT="/home/junyao/LfHV/r3m/eval_offline_checkpoints"
export SAVE="widowx_during_interaction"

echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "N_EVAL_SAMPLES: ${N_EVAL_SAMPLES}"
echo "DATASET: ${DATASET}"
echo "IOU_THRESH: ${IOU_THRESH}"
echo "CKPT_NUM: ${CKPT_NUM}"
echo "EVALTIME: ${EVALTIME}"
echo "DATA_HOME_DIR: ${DATA_HOME_DIR}"
echo "CHECKPOINT: ${CHECKPOINT}"
echo "SAVE: ${SAVE}"

CUDA_VISIBLE_DEVICES=0 xvfb-run -a python /home/junyao/LfHV/r3m/eval_bc.py \
--run_on_cv_server \
--num_workers=${NUM_WORKERS} \
--n_eval_samples=${N_EVAL_SAMPLES} \
--use_visualizer \
--dataset=${DATASET} \
--iou_thresh=${IOU_THRESH} \
--no_task_labels --no_future_labels \
--eval_tasks \
--log_depth --log_depth_scatter_plots --log_metric \
--use_current_frame_info --no_shuffle \
--checkpoint=${CHECKPOINT} \
--root=${ROOT} \
--save=${SAVE} \
--data_home_dir=${DATA_HOME_DIR} \
