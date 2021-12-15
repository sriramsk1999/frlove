"""
Pseudolabel target dataset
"""

import argparse
import fasttext


def pseudolabel_data(model, dataset):
    """Soft-label target dataset with base classes."""
    num_labels = len(model.labels)
    return model.predict(dataset, k=num_labels)


def main(args):
    """Main"""

    # Load teacher model
    model = fasttext.load_model(f"teacher-{args.teacher}.bin")

    # Load target (unalabeled) data
    filepath = f"data/target/unlabeled_{args.dataset}.txt"
    with open(filepath, "r", encoding="utf-8") as datafile:
        dataset = datafile.read().split("\n")

    # Pseudolabel dataset using teacher model
    pseudolabels = pseudolabel_data(model, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pseudolabel target dataset using teacher model."
    )
    parser.add_argument(
        "--teacher",
        help="language of teacher model to use in pseudolabel e.g. French ='fr' ",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        help="target dataset to use in pseudolabel e.g. English ='en' ",
        required=True,
    )
    arguments = parser.parse_args()
    main(arguments)
