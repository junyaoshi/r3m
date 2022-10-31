# command line flags
while getopts n:t:m:c:d: flag
do
    case "${flag}" in
        n) n_blocks=${OPTARG};;
        t) net_type=${OPTARG};;
        m) pred_mode=${OPTARG};;
        c) cuda_device=${OPTARG};;
        d) date_time=${OPTARG};;
        *) echo "usage: $0 [-n] [-t] [-m]" >&2
           exit 1 ;;
    esac
done

export N_BLOCKS=$n_blocks; echo "N_BLOCKS: ${N_BLOCKS}"
export NET_TYPE=$net_type; echo "NET_TYPE: ${NET_TYPE}"
export LR=0.001; echo "LR: ${LR}"
export LAMBDA1=1; echo "LAMBDA1: ${LAMBDA1}"
export LAMBDA2=1; echo "LAMBDA2: ${LAMBDA2}"
export LAMBDA3=1; echo "LAMBDA3: ${LAMBDA3}"
export LAMBDA4=1; echo "LAMBDA4: ${LAMBDA4}"
export PRED_NODE=$pred_mode; echo "PRED_NODE: ${PRED_NODE}"
export EVAL_FREQ=100; echo "EVAL_FREQ: ${EVAL_FREQ}"
export SAVE_FREQ=2000; echo "SAVE_FREQ: ${SAVE_FREQ}"
export EPOCHS=2000; echo "EPOCHS: ${EPOCHS}"
export SANITY_CHECK_SIZE=5; echo "SANITY_CHECK_SIZE: ${SANITY_CHECK_SIZE}"
export BATCH_SIZE=5; echo "BATCH_SIZE: ${BATCH_SIZE}"
export VIS_SAMPLE_SIZE=5; echo "VIS_SAMPLE_SIZE: ${VIS_SAMPLE_SIZE}"
export TASK_VIS_SAMPLE_SIZE=0; echo "TASK_VIS_SAMPLE_SIZE: ${TASK_VIS_SAMPLE_SIZE}"
export NUM_WORKERS=4; echo "NUM_WORKERS: ${NUM_WORKERS}"
export DATETIME=$date_time; echo "DATETIME: ${DATETIME}"
export SAVE="sanity_check/${DATETIME}/cv_net=${NET_TYPE}_nblocks=${N_BLOCKS}_pred=${PRED_NODE}_date=${DATETIME}"
echo "SAVE: ${SAVE}"

CUDA_VISIBLE_DEVICES=$cuda_device xvfb-run -a python /home/junyao/LfHV/r3m/train_transferable_bc.py \
--n_blocks=${N_BLOCKS} \
--net_type=${NET_TYPE} \
--lr=${LR} \
--lambda1=${LAMBDA1} \
--lambda2=${LAMBDA2} \
--lambda3=${LAMBDA3} \
--lambda4=${LAMBDA4} \
--pred_mode=${PRED_NODE} \
--eval_freq=${EVAL_FREQ} \
--save_freq=${SAVE_FREQ} \
--epochs=${EPOCHS} \
--sanity_check \
--sanity_check_size=${SANITY_CHECK_SIZE} \
--batch_size=${BATCH_SIZE} \
--vis_sample_size=${VIS_SAMPLE_SIZE} \
--task_vis_sample_size=${TASK_VIS_SAMPLE_SIZE} \
--num_workers=${NUM_WORKERS} \
--run_on_cv_server \
--use_visualizer \
--save=${SAVE} \
