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

If you want to merge image with prediction, you can add a `--merge-image` flag:
```bash
python tools/demo.py --cfg configs/CONFIG_FILE \
    --input-files PATH_TO_INPUT_FILES \
    --output-dir PATH_TO_OUTPUT_DIR \
    --merge-image \
    TEST.MODEL_FILE YOUR_DOWNLOAD_MODEL_FILE
```

### Training & Evaluation in Command Line
We provide a script in "tools/train_net.py", that is made to train
all the configs provided in panoptic-deeplab.

To train a model with "train_net.py", first
setup the corresponding datasets following
[datasets/README.md](https://github.com/bowenc0221/panoptic-deeplab/blob/master/datasets/README.md),
then run the following command with NUM_GPUS gpus:
```bash
python -m torch.distributed.launch --nproc_per_node=NUM_GPUs tools/train_net.py --cfg configs/CONFIG_FILE
```

To train a model with a single GPU:
```bash
python tools/train_net.py --cfg configs/CONFIG_FILE TRAIN.IMS_PER_BATCH 1 GPUS '(0, )'
```

To evaluate a model's performance, use
```bash
python tools/test_net_single_core.py --cfg configs/CONFIG_FILE
```

To evaluate a model with test time augmentation (e.g. flip and multi-scale), use
```bash
python tools/test_net_single_core.py --cfg configs/CONFIG_FILE \
    TEST.TEST_TIME_AUGMENTATION True \
    TEST.FLIP_TEST True \
    TEST.SCALE_LIST '[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2]'
```
