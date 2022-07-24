export NUM_WORKERS=4
export N_EVAL_SAMPLES=200
export DATASET="ss"
export DATA_HOME_DIR="/home/junyao/Datasets/something_something_processed"
export CHECKPOINT="/scratch/junyao/LfHV/r3m/checkpoints/transfer/cluster_model=transfer_blocks=4_net=residual_\
time=20_lr=0.0004_lambdas=[1,5,0]_batch=64_date=06261730/checkpoint_0300.pt"
export SAVE="cluster_model=transfer_blocks=4_net=residual_\
time=20_lr=0.0004_lambdas=[1,5,0]_batch=64_date=06261730_checkpoint_0300_evaldata=${DATASET}_evaltime=07220130"

echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "N_EVAL_SAMPLES: ${N_EVAL_SAMPLES}"
echo "DATASET: ${DATASET}"
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
--eval_tasks \
--log_depth \
--log_depth_scatter_plots