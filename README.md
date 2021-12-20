FrLove: Could a Frenchman rapidly learn to identify Lovecraft?

Based on STARTUP

1. Create a virtual environment and download the following libraries.

- torch
- pandas
- sklearn
- tqdm
- wget
- textblob
- fasttext
- gensim==3.8.3
- textaugment
- gutenberg

2. Download raw data and process it (Prepare base for French and German, target for English)

```sh
python prepare_data.py --base fr --target en
python prepare_data.py --base de --target en
```

3. Train a teacher model, used for pseudolabeling in STARTUP (train a french teacher)

```sh
python train_teacher.py --base fr
```

4. Run STARTUP (WIP)
