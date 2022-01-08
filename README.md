# FrLove: Could a Frenchman rapidly learn to identify Lovecraft?

**Based on [STARTUP](https://openreview.net/forum?id=O3Y56aqpChA)**

1. Create a virtual environment and download the following libraries.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Download raw data and process it (Prepare base for French, target for English)

```sh
python prepare_data.py --base fr --target en --n_base 5
```

3. Train a teacher model, used for pseudolabeling in STARTUP (train french teacher)

```sh
python train_teacher.py --lang fr
```

4. Run STARTUP - train student model

```sh
python startup.py --base fr --target en --n_base 5
```

5. Evaluate STARTUP and Naive Transfer

```sh
python finetune.py --base fr --target en --n_way 5 --n_base 5
python finetune.py --embedding_load_path student_fr_en_best.pkl --base fr --target en --n_way 5 --n_base 5
```
