# ------------------------------------------------------------------------------
# Utility functions.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def get_loss_info_str(loss_meter_dict):
    msg = ''
    for key in loss_meter_dict.keys():
        msg += '{name}: {meter.val:.3e} ({meter.avg:.3e})\t'.format(
            name=key, meter=loss_meter_dict[key]
        )

    return msg
