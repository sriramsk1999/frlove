import argparse
import os
import random

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
import fasttext

from sklearn.model_selection import train_test_split


class projector_SIMCLR(nn.Module):
    """
    The projector for SimCLR. This is added on top of a backbone for SimCLR Training
    """

    def __init__(self, in_dim, out_dim):
        super(projector_SIMCLR, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def pseudolabel_data(model, dataset):
    """Soft-label target dataset with base classes."""
    num_labels = len(model.labels)
    pseudolabels = model.predict(dataset, k=num_labels)
    # Convert to list of tuples [(data, labels, probs), ...]
    pseudolabeled_data = list(zip(dataset, *pseudolabels))
    return pseudolabeled_data


def seed_everything(seed):
    """Seed everything for reproducibility"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_datasets(model, base, target):
    """Load base and target datasets."""

    # Load base dataset
    base_datafile = f"data/base/{base}.txt"
    with open(base_datafile, "r", encoding="utf-8") as datafile:
        base_dataset = datafile.read().split("\n")
    # Separate label and data
    base_dataset = [(b.split(" ")[0], " ".join(b.split(" ")[1:])) for b in base_dataset]

    # Load target dataset
    target_datafile = f"data/base/{target}.txt"
    with open(target_datafile, "r", encoding="utf-8") as datafile:
        target_dataset = datafile.read().split("\n")

    # Pseudolabel target dataset
    target_dataset = pseudolabel_data(model, target_dataset)

    return base_dataset, target_dataset


def create_dataloader(dataset, args):
    """Create dataloaders from dataset"""
    trainset, valset = train_test_split(dataset, test_size=0.1)
    trainloader = DataLoader(
        trainset,
        batch_size=args.bsize,
        num_workers=args.num_workers,
        shuffle=True,
    )
    valloader = DataLoader(
        valset,
        batch_size=args.bsize,
        num_workers=args.num_workers,
        shuffle=True,
    )
    return trainloader, valloader


def main(args):
    """Main"""

    ###########################
    # SETUP
    ###########################

    if not os.path.isdir(args.dir):
        os.makedirs(args.dir)

    # seed the random number generator
    seed_everything(args.seed)

    ###########################
    # Create Models
    ###########################

    # Load Teacher
    model = fasttext.load_model(f"teacher-{args.base}.bin")
    feature_dim = model.dim
    num_classes = model.labels

    # the student classifier head
    clf = nn.Linear(feature_dim, num_classes).cuda()

    # projection head for SimCLR
    clf_SIMCLR = projector_SIMCLR(feature_dim, args.projection_dim).cuda()

    ###########################
    # Prepare data
    ###########################

    # Datasets
    base_dataset, target_dataset = load_datasets(model, args.base, args.target)

    # DataLoaders
    trainloader, valloader = create_dataloader(target_dataset, args)
    base_trainloader, base_valloader = create_dataloader(base_dataset, args)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STARTUP")
    parser.add_argument("--seed", type=int, default=42, help="Seed for randomness")
    parser.add_argument(
        "--projection_dim",
        type=int,
        default=128,
        help="Projection Dimension for SimCLR",
    )
    parser.add_argument(
        "--base",
        help="language of base dataset/teacher model e.g. French ='fr' ",
        required=True,
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of workers for dataloader"
    )
    parser.add_argument("--bsize", type=int, default=16, help="batch_size for STARTUP")
    arguments = parser.parse_args()
    main(arguments)
