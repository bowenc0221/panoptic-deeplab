## Getting Started with Panoptic-DeepLab

### Training & Evaluation in Command Line
We provide a script in "tools/train_net.py", that is made to train
all the configs provided in panoptic-deeplab.

To train a model with "train_net.py", first
setup the corresponding datasets following
[datasets/README.md](https://github.com/bowenc0221/panoptic-deeplab/blob/master/datasets/README.md),
then run:
```bash
python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py --cfg configs/Base-Panoptic-DeepLab.yaml
```

To evaluate a model's performance, use
```bash
python tools/test_net_single_core.py --cfg configs/Base-Panoptic-DeepLab.yaml
```