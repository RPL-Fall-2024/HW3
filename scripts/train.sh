# Examples:
# bash scripts/train.sh SEED_ID
# bash scripts/train.sh 0

save_ckpt=True
eval_only=False
config_name=train_rl_lowdim_workspace
seed=${1:-0}
run_dir=/tmp/data/outputs/$(date +%Y-%m-%d_%H-%M-%S)-${config_name}

export HYDRA_FULL_ERROR=1 
python train.py \
    --config-dir=./rl_policy/config/ \
    --config-name=${config_name}.yaml \
    hydra.run.dir=${run_dir} \
    training.seed=${seed} \
    training.device="cuda:0" \
    +checkpoint.save_ckpt=${save_ckpt}
