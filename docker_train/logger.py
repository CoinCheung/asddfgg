#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os
import os.path as osp
import time
import logging

import torch.distributed as dist


def setup_logger(model_type, logpth):
    if dist.is_initialized() and dist.get_rank() != 0:
        log_level = logging.WARNING
        return
    if not osp.exists(logpth): os.makedirs(logpth)
    logfile = '{}-{}.log'.format(model_type, time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = osp.join(logpth, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())
