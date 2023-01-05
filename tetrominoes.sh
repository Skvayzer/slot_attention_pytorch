#!/bin/bash -l
#SBATCH --job-name=quantised_sa_od_tetrominoes_end_to_end
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
##SBATCH --time=0-0:05:00
#SBATCH --partition=titan_X
##SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --mail-user=k.smirnov@innopolis.university
#SBATCH --mail-type=END
#SBATCH --no-kill
#SBATCH --comment="Обучение нейросетевой модели в рамках НИР Центра когнитивного моделирования"

singularity instance start \
                     --nv  \
                     --bind /home/AI/yudin.da/smirnov_cv/:/home/smirnov_cv/ \
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
      cd /home/smirnov_cv;
      python3 slot_attention_pytorch/train.py --mode "tetrominoes" --path_to_dataset "/home/smirnov_cv/quantised_sa/datasets/multi_objects/tetrominoes" --device 0 --batch_size 64 --max_epochs 534 --seed 1
      free -m;
";

singularity instance stop ml_env