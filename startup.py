""" STARTUP for text. Trains and saves a student model. """
import argparse
import os
import copy
import time
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import fasttext
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import nltk
import textaugment

from util_classes import (
    FastTextEmbeddingBag,
    NTXentLoss,
    projector_SIMCLR,
    GutenbergLangDataset,
)
import utils


def pseudolabel_data(model, dataset):
    """Soft-label target dataset with base classes."""
    num_labels = len(model.labels)
    pseudolabels = model.predict(dataset, k=num_labels)
    # Convert to list of tuples [(labels, probs), ...]
    pseudolabels = list(zip(*pseudolabels))
    # Zip together labels and probs [ [(label, prob)...], ... ].
    # Sort each pseudolabel as LabelEncoder integer-encodes labels in sorted order
    # Therefore, probabilities are expected in the same order.
    pseudolabels = [sorted(zip(*p)) for p in pseudolabels]
    # Extract probabilities and form dataset [(data, probs), ...]
    pseudolabel_probs = [[p[q][1] for q in range(len(p))] for p in pseudolabels]
    pseudolabeled_data = list(zip(dataset, pseudolabel_probs))
    return pseudolabeled_data


def load_datasets(model, base, target, augment1, augment2=None):
    """Load base and target datasets."""

    # Load base dataset
    base_datafile = f"data/base/{base}.txt"
    with open(base_datafile, "r", encoding="utf-8") as datafile:
        base_dataset = datafile.read().split("\n")
    # Separate label and data
    base_labels = [b.split(" ")[0] for b in base_dataset]
    base_data = [" ".join(b.split(" ")[1:]) for b in base_dataset]
    # Integer encode labels
    le = LabelEncoder()
    base_labels = le.fit_transform(base_labels)
    # classes = le.classes_ # Fetch classes
    # Join base_dataset as [(data, label), ...]
    base_dataset = list(zip(base_data, base_labels))

    # Load target dataset
    target_datafile = f"data/target/unlabeled_{target}.txt"
    with open(target_datafile, "r", encoding="utf-8") as datafile:
        target_dataset = datafile.read().split("\n")

    # Pseudolabel target dataset
    target_dataset = pseudolabel_data(model, target_dataset)
    target_dataset = GutenbergLangDataset(target_dataset, augment1, augment2)

    return base_dataset, target_dataset


def create_dataloader(dataset, args):
    """Create dataloaders from dataset"""
    trainset, valset = train_test_split(dataset, test_size=0.1)
    trainloader = DataLoader(
        trainset,
        batch_size=args.bsize,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,  # SimCLR bug with last smaller batch
    )
    valloader = DataLoader(
        valset,
        batch_size=args.bsize,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,  # SimCLR bug with last smaller batch
    )
    return trainloader, valloader


def checkpoint(model, clf, clf_SIMCLR, optimizer, scheduler, save_path, epoch):
    """
    epoch: the number of epochs of training that has been done
    Should resume from epoch
    """
    sd = {
        "model": copy.deepcopy(model.weight),
        "clf": copy.deepcopy(clf.state_dict()),
        "clf_SIMCLR": copy.deepcopy(clf_SIMCLR.state_dict()),
        "opt": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "epoch": epoch,
    }

    torch.save(sd, save_path)
    return sd


def train(
    model,
    clf,
    clf_SIMCLR,
    optimizer,
    trainloader,
    base_trainloader,
    criterion_SIMCLR,
    epoch,
    num_epochs,
    logger,
    trainlog,
    args,
):

    meters = utils.AverageMeterSet()
    model.train()
    clf.train()
    clf_SIMCLR.train()

    kl_criterion = nn.KLDivLoss(reduction="batchmean")
    nll_criterion = nn.NLLLoss(reduction="mean")

    base_loader_iter = iter(base_trainloader)

    end = time.time()
    for i, ((X1, X2), y) in tqdm(enumerate(trainloader)):
        meters.update("Data_time", time.time() - end)

        current_lr = optimizer.param_groups[0]["lr"]
        meters.update("lr", current_lr, 1)

        y = y.cuda()

        # Get the data from the base dataset
        try:
            X_base, y_base = base_loader_iter.next()
        except StopIteration:
            base_loader_iter = iter(base_trainloader)
            X_base, y_base = base_loader_iter.next()

        y_base = y_base.cuda()

        optimizer.zero_grad()

        # cross entropy loss on the base dataset
        features_base = model(X_base).cuda()
        logits_base = clf(features_base)
        log_probability_base = F.log_softmax(logits_base, dim=1)
        loss_base = nll_criterion(log_probability_base, y_base)

        f1 = model(X1).cuda()
        f2 = model(X2).cuda()

        # SIMCLR Loss on the target dataset
        z1 = clf_SIMCLR(f1)
        z2 = clf_SIMCLR(f2)

        loss_SIMCLR = criterion_SIMCLR(z1, z2)

        # Pseudolabel loss on the target dataset
        logits_xtask_1 = clf(f1).cuda()
        logits_xtask_2 = clf(f2).cuda()
        log_probability_1 = F.log_softmax(logits_xtask_1, dim=1)
        log_probability_2 = F.log_softmax(logits_xtask_2, dim=1)

        loss_xtask = (
            kl_criterion(log_probability_1, y) + kl_criterion(log_probability_2, y)
        ) / 2

        loss = loss_base + loss_SIMCLR + loss_xtask

        loss.backward()
        optimizer.step()

        meters.update("Loss", loss.item(), 1)
        meters.update("KL_Loss_target", loss_xtask.item(), 1)
        meters.update("CE_Loss_source", loss_base.item(), 1)
        meters.update("SIMCLR_Loss_target", loss_SIMCLR.item(), 1)

        perf = utils.accuracy(logits_xtask_1.data, y.argmax(dim=1).data, topk=(1,))

        meters.update("top1", perf["average"][0].item(), len(X1))
        meters.update("top1_per_class", perf["per_class_average"][0].item(), 1)

        perf_base = utils.accuracy(logits_base.data, y_base.data, topk=(1,))

        meters.update("top1_base", perf_base["average"][0].item(), len(X_base))

        meters.update(
            "top1_base_per_class", perf_base["per_class_average"][0].item(), 1
        )

        meters.update("Batch_time", time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            values = meters.values()
            averages = meters.averages()
            sums = meters.sums()

            logger_string = (
                "Training Epoch: [{epoch}/{epochs}] Step: [{step} / {steps}] "
                "Batch Time: {meters[Batch_time]:.4f} "
                "Data Time: {meters[Data_time]:.4f} Average Loss: {meters[Loss]:.4f} "
                "Average KL Loss (Target): {meters[KL_Loss_target]:.4f} "
                "Average SimCLR Loss (Target): {meters[SIMCLR_Loss_target]:.4f} "
                "Average CE Loss (Source): {meters[CE_Loss_source]: .4f} "
                "Learning Rate: {meters[lr]:.4f} "
                "Top1: {meters[top1]:.4f} "
                "Top1_per_class: {meters[top1_per_class]:.4f} "
                "Top1_base: {meters[top1_base]:.4f} "
                "Top1_base_per_class: {meters[top1_base_per_class]:.4f} "
            ).format(
                epoch=epoch,
                epochs=num_epochs,
                step=i + 1,
                steps=len(trainloader),
                meters=meters,
            )

            logger.info(logger_string)

    logger_string = (
        "Training Epoch: [{epoch}/{epochs}] Step: [{step}] Batch Time: {meters[Batch_time]:.4f} "
        "Data Time: {meters[Data_time]:.4f} Average Loss: {meters[Loss]:.4f} "
        "Average KL Loss (Target): {meters[KL_Loss_target]:.4f} "
        "Average SimCLR Loss (Target): {meters[SIMCLR_Loss_target]:.4f} "
        "Average CE Loss (Source): {meters[CE_Loss_source]: .4f} "
        "Learning Rate: {meters[lr]:.4f} "
        "Top1: {meters[top1]:.4f} "
        "Top1_per_class: {meters[top1_per_class]:.4f} "
        "Top1_base: {meters[top1_base]:.4f} "
        "Top1_base_per_class: {meters[top1_base_per_class]:.4f} "
    ).format(epoch=epoch + 1, epochs=num_epochs, step=0, meters=meters)

    logger.info(logger_string)

    values = meters.values()
    averages = meters.averages()
    sums = meters.sums()

    trainlog.record(epoch + 1, {**values, **averages, **sums})

    return averages


def validate(
    model,
    clf,
    clf_simclr,
    testloader,
    base_loader,
    criterion_SIMCLR,
    epoch,
    num_epochs,
    logger,
    testlog,
    args,
    postfix="Validation",
):
    meters = utils.AverageMeterSet()
    model.eval()
    clf.eval()
    clf_simclr.eval()

    criterion_xtask = nn.KLDivLoss(reduction="batchmean")
    nll_criterion = nn.NLLLoss(reduction="mean")

    logits_xtask_test_all = []
    losses_simclr = []
    ys_all = []

    end = time.time()
    # Compute the loss for the target dataset
    with torch.no_grad():
        for _, ((Xtest, Xrand), y) in enumerate(testloader):
            y = y.cuda()

            ftest = model(Xtest).cuda()
            frand = model(Xrand).cuda()

            ztest = clf_simclr(ftest)
            zrand = clf_simclr(frand)

            # get the logits for xtask
            logits_xtask_test = clf(ftest)
            logits_xtask_test_all.append(logits_xtask_test)
            ys_all.append(y)

            losses_simclr.append(criterion_SIMCLR(ztest, zrand))

    ys_all = torch.cat(ys_all, dim=0)
    logits_xtask_test_all = torch.cat(logits_xtask_test_all, dim=0)

    log_probability = F.log_softmax(logits_xtask_test_all, dim=1)

    loss_xtask = criterion_xtask(log_probability, ys_all)

    loss_SIMCLR = sum(losses_simclr) / len(losses_simclr)

    logits_base_all = []
    ys_base_all = []
    with torch.no_grad():
        # Compute the loss on the source base dataset
        for X_base, y_base in base_loader:
            y_base = y_base.cuda()

            features = model(X_base).cuda()
            logits_base = clf(features)

            logits_base_all.append(logits_base)
            ys_base_all.append(y_base)

    ys_base_all = torch.cat(ys_base_all, dim=0)
    logits_base_all = torch.cat(logits_base_all, dim=0)

    log_probability_base = F.log_softmax(logits_base_all, dim=1)

    loss_base = nll_criterion(log_probability_base, ys_base_all)

    loss = loss_xtask + loss_SIMCLR + loss_base

    meters.update("CE_Loss_source_test", loss_base.item(), 1)
    meters.update("KL_Loss_target_test", loss_xtask.item(), 1)
    meters.update("SIMCLR_Loss_target_test", loss_SIMCLR.item(), 1)
    meters.update("Loss_test", loss.item(), 1)

    perf = utils.accuracy(
        logits_xtask_test_all.data, ys_all.argmax(dim=1).data, topk=(1,)
    )

    meters.update("top1_test", perf["average"][0].item(), 1)
    meters.update("top1_test_per_class", perf["per_class_average"][0].item(), 1)

    perf_base = utils.accuracy(logits_base_all.data, ys_base_all.data, topk=(1,))

    meters.update("top1_base_test", perf_base["average"][0].item(), 1)
    meters.update(
        "top1_base_test_per_class", perf_base["per_class_average"][0].item(), 1
    )

    meters.update("Batch_time", time.time() - end)

    logger_string = (
        "{postfix} Epoch: [{epoch}/{epochs}]  Batch Time: {meters[Batch_time]:.4f} "
        "Average Test Loss: {meters[Loss_test]:.4f} "
        "Average Test KL Loss (Target): {meters[KL_Loss_target_test]: .4f} "
        "Average Test SimCLR Loss (Target): {meters[SIMCLR_Loss_target_test]: .4f} "
        "Average CE Loss (Source): {meters[CE_Loss_source_test]: .4f} "
        "Top1_test: {meters[top1_test]:.4f} "
        "Top1_test_per_class: {meters[top1_test_per_class]:.4f} "
        "Top1_base_test: {meters[top1_base_test]:.4f} "
        "Top1_base_test_per_class: {meters[top1_base_test_per_class]:.4f} "
    ).format(postfix=postfix, epoch=epoch, epochs=num_epochs, meters=meters)

    logger.info(logger_string)

    values = meters.values()
    averages = meters.averages()
    sums = meters.sums()

    testlog.record(epoch, {**values, **averages, **sums})

    if postfix != "":
        postfix = "_" + postfix

    return averages


def main(args):
    """Main"""

    ###########################
    # SETUP
    ###########################

    if not os.path.isdir(args.dir):
        os.makedirs(args.dir)

    logger = utils.create_logger(os.path.join(args.dir, "checkpoint.log"), __name__)
    trainlog = utils.savelog(args.dir, "train")
    vallog = utils.savelog(args.dir, "val")

    # seed the random number generator
    utils.seed_everything(args.seed)

    # Download resources for text transformations
    nltk.download(["punkt", "wordnet", "averaged_perceptron_tagger"])

    # word2vec_file = "GoogleNews-vectors-negative300.bin.gz"
    # if not os.path.isfile(word2vec_file):
    #     wget.download(f"https://s3.amazonaws.com/dl4j-distribution/{word2vec_file}")
    # word2vec = gensim.models.KeyedVectors.load_word2vec_format(
    #     f"{word2vec_file}", binary=True
    # )
    # word2vec = textaugment.Word2vec(model=word2vec, runs=2)
    wordnet = textaugment.Wordnet(runs=3, n=True)

    ###########################
    # Create Models
    ###########################

    # Load Teacher
    model = fasttext.load_model(f"teacher-{args.base}-{args.n_base}.bin")
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

    # Datasets - Word2vec removed as it is slow
    base_dataset, target_dataset = load_datasets(
        model,
        args.base,
        args.target,
        wordnet.augment,  # word2vec.augment
    )

    # DataLoaders
    trainloader, valloader = create_dataloader(target_dataset, args)
    base_trainloader, base_valloader = create_dataloader(base_dataset, args)

    ############################
    # Loss Function and Optimizer
    ############################

    criterion = NTXentLoss("cuda", args.bsize, args.temp, True)

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
    best_loss = math.inf

    ############################
    # Train
    ############################
    for epoch in range(args.epochs):
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
                os.path.join(
                    args.dir, f"student_{args.base}_{args.n_base}_{args.target}_{epoch + 1}.pkl"
                ),
                epoch + 1,
            )

        if (epoch + 1) % args.eval_freq == 0:
            performance_val = validate(
                backbone,
                clf,
                clf_SIMCLR,
                valloader,
                base_valloader,
                criterion,
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
                    os.path.join(
                        args.dir, f"student_{args.base}_{args.n_base}_{args.target}_best.pkl"
                    ),
                    best_epoch,
                )
                logger.info(f"*** Best model checkpointed at Epoch {best_epoch}")
                best_loss = loss_val
        print(f"Epoch {epoch} complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STARTUP")
    parser.add_argument(
        "--dir", type=str, default=".", help="directory to save the checkpoints"
    )
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
        "--print_freq",
        type=int,
        default=5,
        help="Frequency (in step per epoch) to print training stats",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=1,
        help="Frequency (in epoch) to evaluate on the val set",
    )
    parser.add_argument(
        "--base",
        help="language of base dataset/teacher model e.g. French ='fr' ",
        required=True,
    )
    parser.add_argument(
        "--n_base",
        help="number of base classes",
        required=True,
    )
    parser.add_argument(
        "--target",
        help="language of target dataset e.g. English ='fr' ",
        required=True,
    )
    arguments = parser.parse_args()
    main(arguments)
