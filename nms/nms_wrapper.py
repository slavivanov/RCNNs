# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from .nms.gpu_nms import gpu_nms
from .nms.cpu_nms import cpu_nms

def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if not force_cpu:
        device_id = 0
        return gpu_nms(dets, thresh, device_id=device_id)
    else:
        return cpu_nms(dets, thresh)