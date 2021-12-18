""" STARTUP for text. Trains and saves a student model. """
import argparse
import os
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
import fasttext
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils import FastTextEmbeddingBag, NTXentLoss, projector_SIMCLR


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
    feature_dim = model.get_dimension()
    num_classes = len(model.labels)

    # Backbone of FastText model
    backbone = FastTextEmbeddingBag(model)

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

    ############################
    # Loss Function and Optimizer
    ############################

    criterion = NTXentLoss("cuda", args.bsize, args.temp, True)
    criterion_val = NTXentLoss("cuda", args.bsize, args.temp, True)

    optimizer = torch.optim.SGD(
        [
            {"params": backbone.parameters()},
            {"params": clf.parameters()},
            {"params": clf_SIMCLR.parameters()},
        ],
        lr=0.1,
        momentum=0.9,
        weight_decay=args.wd,
        nesterov=False,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        verbose=False,
        cooldown=10,
        threshold_mode="rel",
        threshold=1e-4,
        min_lr=1e-5,
    )

    ############################
    # Train
    ############################
    for epoch in tqdm(range(args.epochs)):
        perf = train(
            backbone,
            clf,
            clf_SIMCLR,
            optimizer,
            trainloader,
            base_trainloader,
            criterion,
            epoch,
            args.epochs,
            logger,
            trainlog,
            args,
        )

        scheduler.step(perf["Loss/avg"])

        if (epoch + 1) % args.save_freq == 0:
            checkpoint(
                backbone,
                clf,
                clf_SIMCLR,
                optimizer,
                scheduler,
                os.path.join(args.dir, f"checkpoint_{epoch + 1}.pkl"),
                epoch + 1,
            )

        if (epoch + 1) % args.eval_freq == 0:
            performance_val = validate(
                backbone,
                clf,
                clf_SIMCLR,
                valloader,
                base_valloader,
                criterion_val,
                epoch + 1,
                args.epochs,
                logger,
                vallog,
                args,
                postfix="Validation",
            )

            loss_val = performance_val["Loss_test/avg"]

            if best_loss > loss_val:
                best_epoch = epoch + 1
                checkpoint(
                    backbone,
                    clf,
                    clf_SIMCLR,
                    optimizer,
                    scheduler,
                    os.path.join(args.dir, "checkpoint_best.pkl"),
                    best_epoch,
                )
                logger.info(f"*** Best model checkpointed at Epoch {best_epoch}")
                best_loss = loss_val


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
        "--num_workers", type=int, default=8, help="Number of workers for dataloader"
    )
    parser.add_argument("--bsize", type=int, default=16, help="batch_size for STARTUP")
    parser.add_argument("--temp", type=float, default=1, help="Temperature of SIMCLR")
    parser.add_argument(
        "--wd", type=float, default=1e-4, help="Weight decay for the model"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--save_freq", type=int, default=5, help="Frequency (in epoch) to save"
    )
    parser.add_argument(
        "--base",
        help="language of base dataset/teacher model e.g. French ='fr' ",
        required=True,
    )
    arguments = parser.parse_args()
    main(arguments)
