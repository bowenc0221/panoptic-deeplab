# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/env.py#L15
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import importlib
import importlib.util
import logging
import numpy as np
import os
import random
import sys
from datetime import datetime
import torch

__all__ = ["seed_all_rng"]


def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.
    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        logger = logging.getLogger(__name__)
        logger.info("Using a generated random seed {}".format(seed))
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)
