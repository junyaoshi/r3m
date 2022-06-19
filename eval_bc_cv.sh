export NUM_WORKERS=4
export N_EVAL_SAMPLES=20
export CHECKPOINT="/home/junyao/LfHV/r3m/checkpoints/cluster_model=resnet32_time=20_lr=0.0004_batch=64_workers=4_date=06152345/checkpoint_0248.pt"
export SAVE="cv_model=resnet32_time=20_date=06152345_ckpt=0248_data=train"

echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "N_EVAL_SAMPLES: ${N_EVAL_SAMPLES}"
echo "CHECKPOINT: ${CHECKPOINT}"
echo "SAVE: ${SAVE}"

CUDA_VISIBLE_DEVICES=1 xvfb-run -a python /home/junyao/LfHV/r3m/eval_bc.py \
--num_workers=${NUM_WORKERS} \
--n_eval_samples=${N_EVAL_SAMPLES} \
--checkpoint=${CHECKPOINT} \
--save=${SAVE} \
--run_on_cv_server \
--use_visualizer \
--eval_on_train