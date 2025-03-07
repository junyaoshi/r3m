#SBATCH --mem-per-gpu=12G
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dineshj-high
#SBATCH --cpus-per-gpu=4
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH -w node-2080ti-5
#SBATCH --job-name=t15n2N
#SBATCH -o /home/junyao/transferable_out/t=15_nblocks=2_lambdas=[1,1,1,1].out

#SBATCH --mem-per-gpu=24G
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dineshj-high
#SBATCH --cpus-per-gpu=4
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH -w node-3090-0
#SBATCH --job-name=n32norm
#SBATCH -o /home/junyao/transferable_out/nblocks=32_normal.out