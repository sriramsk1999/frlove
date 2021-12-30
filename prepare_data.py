"""
Prepare Project Gutenberg data
"""
import os
import argparse
import json
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from textblob import TextBlob
import nltk
from sklearn.model_selection import train_test_split


def download_raw_data():
    """
    Download raw data from Project Gutenberg
    """
    if not os.path.isdir("data/raw"):
        os.mkdir("data/raw")

    # Books to be downloaded are stored in data/sources.json
    # Author code mappings are stored in data/author_info.json
    with open("data/sources.json", "r", encoding="utf-8") as source_file:
        source = json.loads(source_file.read())

    for lang, values in source.items():
        if not os.path.isdir(f"data/raw/{lang}"):
            os.mkdir(f"data/raw/{lang}")
        for author, books in values.items():
            print(f"Downloading books of {author}")
            for book in books:
                TXT = strip_headers(load_etext(book)).strip()
                with open(
                    f"data/raw/{lang}/{author}-{book}.txt", "w", encoding="utf-8"
                ) as dest_file:
                    dest_file.write(TXT)
        print(f"Downloaded data for {lang}!")


def create_base_dataset(lang):
    """Create base dataset in FastText format."""

    if not os.path.isdir(f"data/raw/{lang}"):
        raise IOError(f"Data for {lang} does not exist in data/raw.")

    data = []
    chunk_size = 10
    # Load all text files in lang/
    for filename in os.listdir(f"data/raw/{lang}"):
        author = filename.split("-")[0]
        with open(f"data/raw/{lang}/{filename}", "r", encoding="utf-8") as txtfile:
            txtblob = TextBlob(txtfile.read())
        # Split book into chunks of sentences
        chunks = [
            txtblob.sentences[i : i + chunk_size]
            for i in range(0, len(txtblob.sentences) - chunk_size, chunk_size)
        ]
        # Process each chunk into a single string and replace \n with ' '.
        # This is because we are reading txt file at 80 characters width, and sentences spill over.
        # Also because FastText separates each training example using newlines
        # and each chunk is a single example.
        chunks = [
            " ".join([str(sent).replace("\n", " ") for sent in chunk])
            for chunk in chunks
        ]
        # Label chunks with author name in FastText format
        chunks = [f"__label__{author} {chunk}" for chunk in chunks]
        data.extend(chunks)
    if not os.path.isdir("data/base"):
        os.mkdir("data/base")
    with open(f"data/base/{lang}.txt", "w", encoding="utf-8") as base_file:
        base_file.write("\n".join(data))


def create_target_dataset(lang):
    """Create target dataset, splitting into labeled and unlabeled data."""

    unlabeled_perc = 0.2

    # Create base dataset and use it to create target dataset
    create_base_dataset(lang)

    with open(f"data/base/{lang}.txt", "r", encoding="utf-8") as txtfile:
        data = txtfile.read()

    data = data.split("\n")
    labeled, unlabeled = train_test_split(
        data, test_size=unlabeled_perc, random_state=42
    )

    # Remove labels for unlabeled i.e. remove first token when splitting by space
    unlabeled = [" ".join(l.split(" ")[1:]) for l in unlabeled]

    if not os.path.isdir("data/target"):
        os.mkdir("data/target")
    # Write data
    with open(f"data/target/labeled_{lang}.txt", "w", encoding="utf-8") as txtfile:
        txtfile.write("\n".join(labeled))
    with open(f"data/target/unlabeled_{lang}.txt", "w", encoding="utf-8") as txtfile:
        txtfile.write("\n".join(unlabeled))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare data from Project Gutenberg."
    )
    parser.add_argument(
        "--base", help="Language for base dataset - [en, fr, de]", required=True
    )
    parser.add_argument(
        "--target",
        help="Language for target dataset - [en, fr, de]",
        required=True,
    )
    args = parser.parse_args()

    nltk.download("punkt")

    download_raw_data()
    print("Raw data download complete.\n------------------------------------")
    create_base_dataset(args.base)
    print(
        f"Base dataset for {args.base} created.\n------------------------------------"
    )
    create_target_dataset(args.target)
    print(
        f"Target dataset for {args.target} created.\n------------------------------------"
    )
