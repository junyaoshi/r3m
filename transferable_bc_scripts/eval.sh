export TIME_INTERVAL=15; echo "TIME_INTERVAL: ${TIME_INTERVAL}"
export IOU_THRESH=0.6; echo "IOU_THRESH: ${IOU_THRESH}"
export BATCH_SIZE=64; echo "BATCH_SIZE: ${BATCH_SIZE}"
export VIS_FREQ=3; echo "VIS_FREQ: ${VIS_FREQ}"
export VIS_SAMPLE_SIZE=5; echo "VIS_SAMPLE_SIZE: ${VIS_SAMPLE_SIZE}"
export TASK_VIS_SAMPLE_SIZE=2; echo "TASK_VIS_SAMPLE_SIZE: ${TASK_VIS_SAMPLE_SIZE}"
export NUM_WORKERS=0; echo "NUM_WORKERS: ${NUM_WORKERS}"

export TRAIN_DATETIME=11082030
export TRAIN_RUNNAME="cluster_t=15_net=residual_nblocks=8_pred=residual_lr=0.0004_lambdas=[10,1,1,10]_bce=1_batch=128"
export CKPT_NUM=0090

#export TRAIN_RUNNAME="cluster_t=15_net=residual_nblocks=8_pred=residual_lr=0.0004_lambdas=[10,1,1,10000]_bce=1e-4_batch=128"
#export CKPT_NUM=0200

export CHECKPOINT="/scratch/junyao/LfHV/r3m/transferable_bc_ckpts/${TRAIN_DATETIME}/${TRAIN_RUNNAME}/checkpoint_${CKPT_NUM}.pt"
echo "CHECKPOINT: ${CHECKPOINT}"

## widowx 9.22 during interaction (paired)
#export EVALDATA_NAME="widowx_9.22_during_interaction_paired"
#export DATA_HOME_DIRS="/home/junyao/WidowX_Datasets/something_something_hand_robot_paired/widowx_during_interaction_0922 \
#/home/junyao/WidowX_Datasets/something_something_hand_robot_paired/widowx_during_interaction_1"

## widowx 9.22 during interaction (hand)
#export EVALDATA_NAME="widowx_9.22_during_interaction_hand"
#export DATA_HOME_DIRS="/home/junyao/WidowX_Datasets/something_something_hand_robot_paired/widowx_during_interaction_0922 \
#/home/junyao/WidowX_Datasets/something_something_hand_robot_paired/widowx_during_interaction_1 \
#/home/junyao/WidowX_Datasets/something_something_hand_robot_paired/hand_only_test"

## widowx 9.22 pre interaction (paired)
#export EVALDATA_NAME="widowx_9.22_pre_interaction_paired"
#export DATA_HOME_DIRS="/home/junyao/WidowX_Datasets/something_something_hand_robot_paired/widowx_pre_interaction"

## widowx 9.22 pre interaction (hand)
#export EVALDATA_NAME="widowx_9.22_pre_interaction_hand"
#export DATA_HOME_DIRS="/home/junyao/WidowX_Datasets/something_something_hand_robot_paired/widowx_pre_interaction \
#/home/junyao/WidowX_Datasets/something_something_pre_interaction_check_online/var_hand_same_object \
#/home/junyao/WidowX_Datasets/something_something_pre_interaction_check_online/same_hand_var_object"

## franka during interaction (paired)
#export EVALDATA_NAME="franka_during_interaction_paired"
#export DATA_HOME_DIRS="/home/junyao/Franka_Datasets/something_something_hand_robot_paired/franka_during_interaction"

## franka during interaction (hand)
#export EVALDATA_NAME="franka_during_interaction_hand"
#export DATA_HOME_DIRS="/home/junyao/Franka_Datasets/something_something_hand_robot_paired/franka_during_interaction \
#/home/junyao/Franka_Datasets/something_something_hand_demos/during_interaction"

## franka pre interaction (paired)
#export EVALDATA_NAME="franka_pre_interaction_paired"
#export DATA_HOME_DIRS="/home/junyao/Franka_Datasets/something_something_hand_robot_paired/franka_pre_interaction"

## franka pre interaction (hand)
#export EVALDATA_NAME="franka_pre_interaction_hand"
#export DATA_HOME_DIRS="/home/junyao/Franka_Datasets/something_something_hand_robot_paired/franka_pre_interaction \
#/home/junyao/Franka_Datasets/something_something_hand_demos/pre_interaction"

## full during interaction (paired)
#export EVALDATA_NAME="all_during_interaction_paired"
#export DATA_HOME_DIRS="/home/junyao/WidowX_Datasets/something_something_hand_robot_paired/widowx_during_interaction_0922 \
#/home/junyao/WidowX_Datasets/something_something_hand_robot_paired/widowx_during_interaction_1 \
#/home/junyao/Franka_Datasets/something_something_hand_robot_paired/franka_during_interaction"

## full during interaction (hand)
#export EVALDATA_NAME="all_during_interaction_hand"
#export DATA_HOME_DIRS="/home/junyao/WidowX_Datasets/something_something_hand_robot_paired/widowx_during_interaction_0922 \
#/home/junyao/WidowX_Datasets/something_something_hand_robot_paired/widowx_during_interaction_1 \
#/home/junyao/Franka_Datasets/something_something_hand_robot_paired/franka_during_interaction \
#/home/junyao/WidowX_Datasets/something_something_hand_robot_paired/hand_only_test \
#/home/junyao/Franka_Datasets/something_something_hand_demos/during_interaction"

# RoboAware
export EVALDATA_NAME="roboaware"
export DATA_HOME_DIRS="/home/junyao/Datasets/something_something_hand_demos"

## widowx_7.18 (cannot evaluate because it only has 4 tasks)
#export EVALDATA_NAME="widowx_7.18"
#export DATA_HOME_DIRS="/home/junyao/Datasets/something_something_robot_demos"

echo "DATA_HOME_DIRS: ${DATA_HOME_DIRS}"
export SAVE="${EVALDATA_NAME}/${TRAIN_DATETIME}_${TRAIN_RUNNAME}_ckpt=${CKPT_NUM}"
echo "SAVE: ${SAVE}"


CUDA_VISIBLE_DEVICES=1 xvfb-run -a python /home/junyao/LfHV/r3m/eval_transferable_bc.py \
--time_interval=${TIME_INTERVAL} \
--iou_thresh=${IOU_THRESH} \
--eval_tasks \
--batch_size=${BATCH_SIZE} \
--vis_freq=${VIS_FREQ} \
--log_depth_scatter_plots \
--num_workers=${NUM_WORKERS} \
--run_on_cv_server \
--use_visualizer \
--checkpoint=${CHECKPOINT} \
--save=${SAVE} \
--data_home_dirs ${DATA_HOME_DIRS} \
--vis_sample_size=${VIS_SAMPLE_SIZE} \
--task_vis_sample_size=${TASK_VIS_SAMPLE_SIZE} \
--has_task_labels --has_future_labels #--eval_robot --no_shuffle