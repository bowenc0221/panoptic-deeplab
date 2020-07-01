# ------------------------------------------------------------------------------
# Generates the correct format for official evaluation code.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from collections import OrderedDict

import numpy as np


def get_cityscapes_instance_format(panoptic, sem, ctr_hmp, label_divisor):
    """
    Get Cityscapes instance segmentation format.
    Arguments:
        panoptic: A Numpy Ndarray of shape [H, W].
        sem: A Numpy Ndarray of shape [C, H, W] of raw semantic output.
        ctr_hmp: A Numpy Ndarray of shape [H, W] of raw center heatmap output.
        label_divisor: An Integer, used to convert panoptic id = semantic id * label_divisor + instance_id.
    Returns:
        A List contains instance segmentation in Cityscapes format.
    """
    instances = []

    pan_labels = np.unique(panoptic)
    for pan_lab in pan_labels:
        if pan_lab % label_divisor == 0:
            # This is either stuff or ignored region.
            continue

        ins = OrderedDict()

        train_class_id = pan_lab // label_divisor
        ins['pred_class'] = train_class_id

        mask = panoptic == pan_lab
        ins['pred_mask'] = np.array(mask, dtype='uint8')

        sem_scores = sem[train_class_id, ...]
        ins_score = np.mean(sem_scores[mask])
        ins['score'] = ins_score

        instances.append(ins)

    return instances
