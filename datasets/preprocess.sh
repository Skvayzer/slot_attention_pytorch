#!/bin/bash -l
#SBATCH --job-name=clevr_with_mask_preprocess
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=GTX780
#SBATCH --mem-per-gpu=16GB
#SBATCH --mail-user=k.smirnov@innopolis.university
#SBATCH --mail-type=END
#SBATCH --no-kill
#SBATCH --comment="Обучение нейросетевой модели в рамках НИР Центра когнитивного моделирования"


singularity instance start \
                     --nv  \
                     --bind /home/AI/yudin.da/smirnov_cv/slot_attention_pytorch/datasets/:/home/sa \
                     /home/AI/yudin.da/smirnov_cv/quantised_sa/ml_env.sif ml_env

singularity exec instance://ml_env /bin/bash -c "
      source /miniconda/etc/profile.d/conda.sh;
      conda activate ml_env;
      export WANDB_API_KEY=c84312b58e94070d15277f8a5d58bb72e57be7fd;
      set -x;
      ulimit -Hn;
      ulimit -Sn;
      nvidia-smi;
      free -m;
      cd /home/sa;
      python3 clevr_with_masks.py;
      free -m;
";

singularity instance stop ml_env