import argparse
import os
import random

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import fasttext

class projector_SIMCLR(nn.Module):
    '''
        The projector for SimCLR. This is added on top of a backbone for SimCLR Training
    '''
    def __init__(self, in_dim, out_dim):
        super(projector_SIMCLR, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

def main(args):

    ###########################
    # SETUP
    ###########################

    if not os.path.isdir(args.dir):
        os.makedirs(args.dir)

    # seed the random number generator
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

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

    # Load base dataset
    base_datafile = f"data/base/{args.base}.txt"
    with open(base_datafile, 'r', encoding='utf-8') as datafile:
        base_dataset = datafile.read().split('\n')

    # Load target dataset
    target_datafile = f"data/base/{args.target}.txt"
    with open(target_datafile, 'r', encoding='utf-8') as datafile:
        target_dataset = datafile.read().split('\n')

    # figure out how to format dataset
    # Pseudolabel target dataset
    # what is that apply_twice bs
    # split data
    # put into torch subset or something
    # put into dataloaders
    # that's the end of this section, dayum we're getting there

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='STARTUP')
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed for randomness')
    parser.add_argument('--projection_dim', type=int, default=128,
                        help='Projection Dimension for SimCLR')
    parser.add_argument(
        "--base",
        help="language of base dataset/teacher model e.g. French ='fr' ",
        required=True,
    )
    arguments = parser.parse_args()
    main(arguments)
