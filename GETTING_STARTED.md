## Getting Started with Panoptic-DeepLab

### Simple demo
Please download pre-trained model from [MODEL_ZOO](MODEL_ZOO.md), replace CONFIG_FILE with
corresponding config file of model you download, and then run

```bash
python tools/demo.py --cfg configs/CONFIG_FILE \
    --input-files PATH_TO_INPUT_FILES \
    --output-dir PATH_TO_OUTPUT_DIR \
    TEST.MODEL_FILE YOUR_DOWNLOAD_MODEL_FILE
```

### Training & Evaluation in Command Line
We provide a script in "tools/train_net.py", that is made to train
all the configs provided in panoptic-deeplab.

To train a model with "train_net.py", first
setup the corresponding datasets following
[datasets/README.md](https://github.com/bowenc0221/panoptic-deeplab/blob/master/datasets/README.md),
then run:
```bash
python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py --cfg configs/CONFIG_FILE
```

To evaluate a model's performance, use
```bash
python tools/test_net_single_core.py --cfg configs/CONFIG_FILE
```
