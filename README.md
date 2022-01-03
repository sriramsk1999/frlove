# FrLove: Could a Frenchman rapidly learn to identify Lovecraft?

**Based on [STARTUP](https://openreview.net/forum?id=O3Y56aqpChA)**

1. Create a virtual environment and download the following libraries.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Download raw data and process it (Prepare base for French and German, target for English)

```sh
python prepare_data.py --base fr --target en
python prepare_data.py --base de --target en
```

3. Train a teacher model, used for pseudolabeling in STARTUP (train french and german teachers)

```sh
python train_teacher.py --lang fr
python train_teacher.py --lang de
```

4. Run STARTUP - train student models

```sh
python startup.py --base fr --target en
python startup.py --base de --target en
```

5. Evaluate STARTUP and Naive Transfer

```sh
python finetune.py --base fr --target en --n_way 5
python finetune.py --embedding_load_path student_fr_en_best.pkl --base fr --target en --n_way 5

python finetune.py --base de --target en --n_way 5
python finetune.py --embedding_load_path student_de_en_best.pkl --base de --target en --n_way 5
```
