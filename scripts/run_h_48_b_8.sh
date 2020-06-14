PYTHON="/opt/conda/bin/python"

CONFIG=$1
# CONFIG="panoptic_deeplab_H48_os4_bs8_cityscapes"

# training
$PYTHON -m torch.distributed.launch \
                --nproc_per_node=8 \
                tools/train_net.py \
                --cfg configs/${CONFIG}.yaml

# evaluation
$PYTHON tools/test_net_single_core.py \
                --cfg configs/${CONFIG}.yaml 