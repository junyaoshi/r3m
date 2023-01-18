# command line flags
while getopts n:s:p:b:c:d: flag
do
    case "${flag}" in
        n) n_blocks=${OPTARG};;
        s) stage=${OPTARG};;
        p) pred_contact=${OPTARG};;
        b) batch_size=${OPTARG};;
        c) cuda_device=${OPTARG};;
        d) date_time=${OPTARG};;
        *) echo "usage: $0 [-n] [-s] [-p] [-b] [-c] [-d]" >&2
           exit 1 ;;
    esac
done

export N_BLOCKS=$n_blocks; echo "N_BLOCKS: ${N_BLOCKS}"
export NET_TYPE=residual; echo "NET_TYPE: ${NET_TYPE}"
export TIME_INTERVAL=10; echo "TIME_INTERVAL: ${TIME_INTERVAL}"
export STAGE=$stage; echo "STAGE: ${STAGE}"
export LR=0.0004; echo "LR: ${LR}"
export LAMBDA1=1; echo "LAMBDA1: ${LAMBDA1}"
export LAMBDA2=1; echo "LAMBDA2: ${LAMBDA2}"
export LAMBDA3=1; echo "LAMBDA3: ${LAMBDA3}"
export LAMBDA4=1; echo "LAMBDA4: ${LAMBDA4}"
export PRED_RESIDUAL=True; echo "PRED_RESIDUAL: ${PRED_RESIDUAL}"
export PRED_CONTACT=${pred_contact}; echo "PRED_CONTACT: ${PRED_CONTACT}"
#export EVAL_FREQ=50; echo "EVAL_FREQ: ${EVAL_FREQ}"
export EVAL_FREQ=10; echo "EVAL_FREQ: ${EVAL_FREQ}"
export SAVE_FREQ=2000; echo "SAVE_FREQ: ${SAVE_FREQ}"
#export EPOCHS=2000; echo "EPOCHS: ${EPOCHS}"
export EPOCHS=500; echo "EPOCHS: ${EPOCHS}"
export SANITY_CHECK_SIZE=$batch_size; echo "SANITY_CHECK_SIZE: ${SANITY_CHECK_SIZE}"
export BATCH_SIZE=$batch_size; echo "BATCH_SIZE: ${BATCH_SIZE}"
export VIS_SAMPLE_SIZE=5; echo "VIS_SAMPLE_SIZE: ${VIS_SAMPLE_SIZE}"
export TASK_VIS_SAMPLE_SIZE=0; echo "TASK_VIS_SAMPLE_SIZE: ${TASK_VIS_SAMPLE_SIZE}"
export NUM_WORKERS=4; echo "NUM_WORKERS: ${NUM_WORKERS}"
export DATETIME=$date_time; echo "DATETIME: ${DATETIME}"
export SAVE="sanity_check/${DATETIME}/cv_nblocks=${N_BLOCKS}_stage=${STAGE}_contact=${PRED_CONTACT}_date=${DATETIME}"
# export SAVE="sanity_check/batchnorm_check_${DATETIME}/bsize=${BATCH_SIZE}_affine=false_forward=false"
#export SAVE="sanity_check/${DATETIME}_implement_check/cv"
echo "SAVE: ${SAVE}"

CUDA_VISIBLE_DEVICES=$cuda_device xvfb-run -a python /home/junyao/LfHV/r3m/train_transferable_bc.py \
--n_blocks=${N_BLOCKS} \
--net_type=${NET_TYPE} \
--time_interval=${TIME_INTERVAL} \
--stage=${STAGE} \
--lr=${LR} \
--lambda1=${LAMBDA1} \
--lambda2=${LAMBDA2} \
--lambda3=${LAMBDA3} \
--lambda4=${LAMBDA4} \
--pred_residual=${PRED_RESIDUAL} \
--pred_contact=${PRED_CONTACT} \
--eval_freq=${EVAL_FREQ} \
--save_freq=${SAVE_FREQ} \
--epochs=${EPOCHS} \
--sanity_check=True \
--sanity_check_size=${SANITY_CHECK_SIZE} \
--eval_tasks=False \
--batch_size=${BATCH_SIZE} \
--vis_sample_size=${VIS_SAMPLE_SIZE} \
--task_vis_sample_size=${TASK_VIS_SAMPLE_SIZE} \
--num_workers=${NUM_WORKERS} \
--run_on_cv_server=True \
--use_visualizer=True \
--save=${SAVE}