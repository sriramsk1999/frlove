""" Finetunes model and runs few-shot evaluation. """
import os
import argparse
import copy

import torch
from torch import nn
import numpy as np
import pandas as pd
from tqdm import tqdm

import utils
from util_classes import Classifier


def finetune(dataloader, params, n_shot):
    """Finetune model on dataset."""
    print("Loading Model: ", params.embedding_load_path)

    state = torch.load(params.embedding_load_path)["state"]
    state_keys = list(state.keys())
    for _, key in enumerate(state_keys):
        if "feature." in key:
            # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
            newkey = key.replace("feature.", "")
            state[newkey] = state.pop(key)
        else:
            state.pop(key)
    sd = state

    n_query = params.n_query
    n_way = params.n_way
    n_support = n_shot

    acc_all = []

    for i, (x, y) in tqdm(enumerate(dataloader)):

        pretrained_model = copy.deepcopy(pretrained_model_template)
        classifier = Classifier(feature_dim, params.n_way)

        pretrained_model.cuda()
        classifier.cuda()

        x = x.cuda()
        x_var = x

        assert len(torch.unique(y)) == n_way

        batch_size = 4
        support_size = n_way * n_support

        y_a_i = torch.from_numpy(np.repeat(range(n_way), n_support)).cuda()

        # split into support and query
        x_b_i = (
            x_var[:, n_support:, :, :, :]
            .contiguous()
            .view(n_way * n_query, *x.size()[2:])
            .cuda()
        )
        x_a_i = (
            x_var[:, :n_support, :, :, :]
            .contiguous()
            .view(n_way * n_support, *x.size()[2:])
            .cuda()
        )  # (25, 3, 224, 224)

        pretrained_model.eval()
        with torch.no_grad():
            f_a_i = pretrained_model(x_a_i)

        loss_fn = nn.CrossEntropyLoss().cuda()
        classifier_opt = torch.optim.SGD(
            classifier.parameters(),
            lr=0.01,
            momentum=0.9,
            dampening=0.9,
            weight_decay=0.001,
        )

        total_epoch = 100
        classifier.train()

        for epoch in range(total_epoch):
            rand_id = np.random.permutation(support_size)

            for j in range(0, support_size, batch_size):
                classifier_opt.zero_grad()

                #####################################
                selected_id = torch.from_numpy(
                    rand_id[j : min(j + batch_size, support_size)]
                ).cuda()

                y_batch = y_a_i[selected_id]

                output = f_a_i[selected_id]
                output = classifier(output)
                loss = loss_fn(output, y_batch)

                #####################################
                loss.backward()

                classifier_opt.step()

        pretrained_model.eval()
        classifier.eval()

        with torch.no_grad():
            output = pretrained_model(x_b_i)
            scores = classifier(output)

        y_query = np.repeat(range(n_way), n_query)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()

        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        correct_this, count_this = float(top1_correct), len(y_query)
        # print (correct_this/ count_this *100)
        acc_all.append((correct_this / count_this * 100))

        if (i + 1) % 100 == 0:
            acc_all_np = np.asarray(acc_all)
            acc_mean = np.mean(acc_all_np)
            acc_std = np.std(acc_all_np)
            print(
                "Test Acc (%d episodes) = %4.2f%% +- %4.2f%%"
                % (len(acc_all), acc_mean, 1.96 * acc_std / np.sqrt(len(acc_all)))
            )

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print(
        "%d Test Acc = %4.2f%% +- %4.2f%%"
        % (len(acc_all), acc_mean, 1.96 * acc_std / np.sqrt(len(acc_all)))
    )

    return acc_all


def main(params):
    """Main"""
    if not os.path.isdir(params.save_dir):
        os.makedirs(params.save_dir)

    results = {}
    shot_done = []
    print(params.target)
    for shot in params.n_shot:
        print(f"{params.n_way}-way {shot}-shot")
        utils.seed_everything(params.seed)
        # what to do about novel_loader? is it just a dataloader? do I need a function for loading data?
        acc_all = finetune(novel_loader, params, n_shot=shot)
        results[shot] = acc_all
        shot_done.append(shot)

        pd.DataFrame(results).to_csv(
            os.path.join(
                params.save_dir,
                params.source_dataset
                + "_"
                + params.target_dataset
                + "_"
                + str(params.n_way)
                + "way_"
                + params.save_suffix
                + ".csv",
            ),
            index=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Few-shot evaluation script")
    parser.add_argument(
        "--save_dir", default=".", type=str, help="Directory to save the result csv"
    )
    parser.add_argument(
        "--n_shot",
        nargs="+",
        default=[5],
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
        "--save_suffix", type=str, required=True, help="suffix added to the csv file"
    )
    parser.add_argument(
        "--embedding_load_path", type=str, required=True, help="path to load embedding"
    )
    parser.add_argument(
        "--base",
        help="language of base dataset e.g. French ='fr' ",
        required=True,
    )
    parser.add_argument(
        "--target",
        help="language of target dataset e.g. English ='fr' ",
        required=True,
    )
    args = parser.parse_args()
    main(args)
