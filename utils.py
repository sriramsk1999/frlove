"""Utility functions"""
import logging
import datetime
import time
import os
import random

import pandas as pd
import torch
import numpy as np


def accuracy(
    logits,
    ground_truth,
    topk=[
        1,
    ],
):
    assert len(logits) == len(ground_truth)
    # this function will calculate per class acc
    # average per class acc and acc

    n, d = logits.shape

    label_unique = torch.unique(ground_truth)
    acc = {}
    acc["average"] = torch.zeros(len(topk))
    acc["per_class_average"] = torch.zeros(len(topk))
    acc["per_class"] = [[] for _ in label_unique]
    acc["gt_unique"] = label_unique
    acc["topk"] = topk
    acc["num_classes"] = d

    max_k = max(topk)
    argsort = torch.argsort(logits, dim=1, descending=True)[:, : min([max_k, d])]
    correct = (argsort == ground_truth.view(-1, 1)).float()

    for indi, i in enumerate(label_unique):
        ind = torch.nonzero(ground_truth == i, as_tuple=False).view(-1)
        correct_target = correct[ind]

        # calculate topk
        for indj, j in enumerate(topk):
            num_correct_partial = torch.sum(correct_target[:, :j]).item()
            acc_partial = num_correct_partial / len(correct_target)
            acc["average"][indj] += num_correct_partial
            acc["per_class_average"][indj] += acc_partial
            acc["per_class"][indi].append(acc_partial * 100)

    acc["average"] = acc["average"] / n * 100
    acc["per_class_average"] = acc["per_class_average"] / len(label_unique) * 100

    return acc


def create_logger(fname, logger_name):
    """Create logger."""
    # Get a logger with name logger_name
    logger = logging.getLogger(logger_name)

    # File handler for log
    hdlr = logging.FileHandler(fname)
    # Format of the logging information
    formatter = logging.Formatter("%(levelname)s %(message)s")

    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)

    # Set the level to logging info, meaning anything information
    # with information level above info will be logged
    logger.setLevel(logging.INFO)

    return logger


class savelog:
    """Saves training log to csv"""

    INCREMENTAL_UPDATE_TIME = 0

    def __init__(self, directory, name):
        self.file_path = os.path.join(
            directory,
            "{}_{:%Y-%m-%d_%H:%M:%S}.csv".format(name, datetime.datetime.now()),
        )
        self.data = {}
        self.last_update_time = time.time() - self.INCREMENTAL_UPDATE_TIME

    def record(self, step, value_dict):
        self.data[step] = value_dict
        if time.time() - self.last_update_time >= self.INCREMENTAL_UPDATE_TIME:
            self.last_update_time = time.time()
            self.save()

    def save(self):
        df = pd.DataFrame.from_dict(self.data, orient="index").to_csv(self.file_path)


class AverageMeterSet:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if not name in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=""):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix="/avg"):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix="/sum"):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix="/count"):
        return {name + postfix: meter.count for name, meter in self.meters.items()}


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        val is the average value
        n : the number of items used to calculate the average
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(
            self=self, format=format
        )


def seed_everything(seed):
    """Seed everything for reproducibility"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
