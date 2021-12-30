""" Finetunes model and runs few-shot evaluation. """
import os
import argparse
import copy
import random

import torch
from torch import nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import fasttext

import utils
from util_classes import Classifier, FastTextEmbeddingBag


def finetune(support_loader, query_loader, params):
    """Finetune and evaluate model on dataset."""
    print("Loading Model: ", params.embedding_load_path)

    # Load teacher into FastTextEmbeddingBag
    pretrained_model = fasttext.load_model(f"teacher-{params.base}.bin")
    feature_dim = pretrained_model.get_dimension()
    pretrained_model = FastTextEmbeddingBag(pretrained_model)
    # Path to STARTUP embedding. Replaces the fasttext embeddings
    # If not set, equivalent to evaluating naive transfer
    if params.embedding_load_path:
        embedding = torch.load(params.embedding_load_path)["model"]
        pretrained_model.weight.data.copy_(embedding)

    pretrained_model.cuda()
    classifier = Classifier(feature_dim, params.n_way).cuda()

    loss_fn = nn.CrossEntropyLoss().cuda()
    classifier_opt = torch.optim.SGD(
        classifier.parameters(),
        lr=0.01,
        momentum=0.9,
        dampening=0.9,
        weight_decay=0.001,
    )

    total_epoch = 20
    pretrained_model.eval()
    classifier.train()

    # Finetune
    for _ in tqdm(range(total_epoch)):
        for x, y in support_loader:
            classifier_opt.zero_grad()

            #####################################

            output = pretrained_model(x)
            output = classifier(output)
            loss = loss_fn(output, y)

            #####################################
            loss.backward()
            classifier_opt.step()

    # Evaluate
    pretrained_model.eval()
    classifier.eval()

    with torch.no_grad():
        for x, y in query_loader:
            output = pretrained_model(x)
            scores = classifier(output)
            # dump scores and labels in one list

    topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
    topk_ind = topk_labels.cpu().numpy()

    top1_correct = np.sum(topk_ind[:, 0] == y_query)
    correct_this, count_this = float(top1_correct), len(y_query)
    acc = correct_this / count_this * 100

    print("Test Acc = %4.2f%%" % (acc))


def create_eval_dataloaders(params):
    """Create dataloaders based on evaluation parameters."""

    # Load target (labeled) dataset
    target_datafile = f"data/target/labeled_{params.target}.txt"
    with open(target_datafile, "r", encoding="utf-8") as datafile:
        target_dataset = datafile.read().split("\n")
    target_dataset = [(i.split()[0], " ".join(i.split()[1:])) for i in target_dataset]
    target_dataset_dict = {}  # Reverse dictionary to store author:[sentences, ..]
    for auth, sents in target_dataset:
        if auth not in target_dataset_dict:
            target_dataset_dict[auth] = []
        target_dataset_dict[auth].append(sents)

    # Extract n_way, n_shot, n_query
    n_items = params.n_shot + params.n_query  # Total number of samples required
    target_dataset_dict = {
        auth: random.sample(sents, n_items)
        for auth, sents in target_dataset_dict.items()
    }

    def create_subset_loader(start, end, dataset):
        subset = {auth: sents[start:end] for auth, sents in dataset.items()}
        subset = [(auth, sents) for sents in subset[auth] for auth in subset.keys()]
        subset_loader = torch.utils.data.DataLoader(
            subset,
            batch_size=params.bsize,
            num_workers=params.num_workers,
            shuffle=True,
        )
        return subset_loader

    support_loader = create_subset_loader(0, params.n_shot, target_dataset_dict)
    query_loader = create_subset_loader(
        params.n_shot, params.n_query, target_dataset_dict
    )
    return support_loader, query_loader


def main(params):
    """Main"""
    if not os.path.isdir(params.save_dir):
        os.makedirs(params.save_dir)

    print("{params.base} -> {params.target}")
    print(f"{params.n_way}-way {params.n_shot}-shot")

    utils.seed_everything(params.seed)
    support_loader, query_loader = create_eval_dataloaders(params)
    finetune(support_loader, query_loader, params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Few-shot evaluation script")
    parser.add_argument(
        "--save_dir", default=".", type=str, help="Directory to save the result csv"
    )
    parser.add_argument(
        "--n_shot",
        default=5,
        type=int,
        help="number of labeled data in each class, same as n_support",
    )
    parser.add_argument(
        "--n_way", default=5, type=int, help="class num to classify for training"
    )
    parser.add_argument(
        "--n_query", default=15, type=int, help="Number of query examples per class"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for randomness")
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of workers for dataloader"
    )
    parser.add_argument("--bsize", type=int, default=16, help="batch_size for STARTUP")
    parser.add_argument(
        "--embedding_load_path",
        type=str,
        help="path to load STARTUP embedding.",
    )
    parser.add_argument(
        "--base",
        help="language of base dataset and teacher model e.g. French ='fr' ",
        required=True,
    )
    parser.add_argument(
        "--target",
        help="language of target dataset e.g. English ='fr' ",
        required=True,
    )
    args = parser.parse_args()
    main(args)
