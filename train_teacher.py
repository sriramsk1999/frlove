"""
Prepare teacher model.
"""

import argparse
import os
import sys
import gzip
import shutil
import fasttext

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

############################### Adapted from download_model in fasttext.util
# https://github.com/facebookresearch/fastText/blob/main/python/fasttext_module/fasttext/util/util.py


def download_vectors(lang_code):
    """Download fasttext vectors of language"""
    file_name = f"cc.{lang_code}.300.vec"
    gz_file_name = f"{file_name}.gz"

    if os.path.isfile(file_name):  # Already exists
        return file_name

    if not os.path.isfile(gz_file_name):
        if _download_gz_model(gz_file_name):
            with gzip.open(gz_file_name, "rb") as f:
                with open(file_name, "wb") as f_out:
                    shutil.copyfileobj(f, f_out)
    return file_name


def _download_gz_model(gz_file_name):
    url = f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/{gz_file_name}"
    chunk_size = 2 ** 13
    print(f"Downloading {url}")
    response = urlopen(url)
    if hasattr(response, "getheader"):
        file_size = int(response.getheader("Content-Length").strip())
    else:
        file_size = int(response.info().getheader("Content-Length").strip())
    downloaded = 0
    download_file_name = gz_file_name + ".part"
    with open(download_file_name, "wb") as f:
        while True:
            chunk = response.read(chunk_size)
            downloaded += len(chunk)
            if not chunk:
                break
            f.write(chunk)
            _print_progress(downloaded, file_size)

    os.rename(download_file_name, gz_file_name)
    return True


def _print_progress(downloaded_bytes, total_size):
    percent = float(downloaded_bytes) / total_size
    bar_size = 50
    bar = int(percent * bar_size)
    percent = round(percent * 100, 2)
    sys.stdout.write(" (%0.2f%%) [" % percent)
    sys.stdout.write("=" * bar)
    sys.stdout.write(">")
    sys.stdout.write(" " * (bar_size - bar))
    sys.stdout.write("]\r")
    sys.stdout.flush()

    if downloaded_bytes >= total_size:
        sys.stdout.write("\n")


############################### Adapted from download_model in fasttext.util


def main(args):
    """Main"""

    # Download FastText vectors
    vecfile = download_vectors(args.lang)

    # Base dataset file
    datafile = f"data/base/{args.lang}.txt"
    if not os.path.isfile(datafile):
        raise IOError(f"Data for {args.lang} does not exist.")

    # Train fasttext supervised model using pretrained vectors
    model = fasttext.train_supervised(
        input=datafile, dim=300, pretrainedVectors=vecfile, epoch=10
    )

    # Save model
    model.save_model(f"teacher-{args.lang}-{len(model.labels)}.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train teacher model on base dataset.")
    parser.add_argument(
        "--lang",
        help="""language of fasttext vectors to use in training
        (lookup language code on https://fasttext.cc/docs/en/crawl-vectors.html#models)
        e.g. French ='fr' """,
        required=True,
    )
    arguments = parser.parse_args()
    main(arguments)
