PYTHON="/opt/conda/bin/python"
$PYTHON -m pip install git+https://github.com/mcordts/cityscapesScripts.git

CONFIG=$1

# training
$PYTHON -m torch.distributed.launch \
                --nproc_per_node=8 \
                tools/train_net.py \
                --cfg configs/${CONFIG}.yaml

# evaluation
$PYTHON tools/test_net_single_core.py \
                --cfg configs/${CONFIG}.yaml
