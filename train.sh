PYTHON="/home/ubuntu/anaconda3/envs/xhy/bin/python"
GPU_NUM=2
CONFIG="seg_hrnet_w48_473x473_sgd_lr7e-3_wd5e-4_bs_40_epoch150.yaml"

$PYTHON -m torch.distributed.launch \
        --nproc_per_node=$GPU_NUM \
        ./tools/train.py \
        --cfg ./experiments/faceparse/$CONFIG.yaml
