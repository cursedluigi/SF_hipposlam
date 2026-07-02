#!/bin/bash
#SBATCH --job-name=first           # Name of your job
#SBATCH --output=output_%j.log      # Output log file (%j is job ID)
#SBATCH --error=error_%j.log        # Error log file
#SBATCH --time=40:00:00             # Time limit (HH:MM:SS)
#SBATCH --partition=cpu        # Partition/queue name
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --cpus-per-task=32           # Number of CPU cores per task
#SBATCH --mem=24G                    # Memory per node
#SBATCH --mail-type=END,FAIL        # When to email you (optional)
#SBATCH --mail-user=khushi.bisani@email.uni-freiburg.de  # Your email address

# Load required modules
#module load python/3.10

# Navigate to your working directory
cd /home/fr/fr_kb470/SF_hipposlam/train_dir
source /home/fr/fr_kb470/miniforge3/etc/profile.d/conda.sh
# eval "$(mamba shell hook --shell bash)"
conda activate KhushiSF

module load vis/mesa/24.1.3-gcccore-13.3.0
module load lib/sdl2/2.30.6-gcccore-13.3.0

# Execute your command
python -m sf_khushi.dmlab.train_hipposlam \
  --env=openfield_map2_fixed_loc3 \
  --experiment=testing_2 \
  --train_for_seconds=140500 \
  --num_workers=2 --num_envs_per_worker=2 \
  --with_pbt=False \
  --with_wandb=True \
  --dmlab_reduced_action_set=True \
  --benchmark=False \
  --algo=APPO --use_rnn=True --rnn_type=gru --core_name=BypassSS --DG_name=linear_relu \
  --Hippo_n_feature=16 --Hippo_L=64 --Hippo_R=8 --rnn_size=1149 \
  --encoder_conv_architecture=layer2_resnet18 \
  --encoder_conv_mlp_layers=256 \
  --depth_sensor=True --dmlab_renderer=software \
  --dmlab_use_level_cache=True \
  --dmlab_one_task_per_worker=True \
  --DG_BN_intercept=2.43 --decoder_mlp_layers 128 128 \
  --env_frameskip=8 --device=cpu --seed=1111 \
  --use_prev_action=True --action_encoding=embedding \
  --action_embed_dim=8