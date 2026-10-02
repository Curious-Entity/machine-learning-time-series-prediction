# App Whisperer: Next-App Prediction

An end-to-end machine-learning project that predicts the next app a person will open from their recent app-usage sequence. The model learns from windows of 30 prior app events and returns a ranked list of likely next apps.

The companion presentation is available at [data-sci-final.lovable.app](https://data-sci-final.lovable.app/).

## Result

The included held-out predictions score **71.81% top-1 accuracy** across **118,345** test sequences. The companion presentation also reports top-3 and top-5 accuracy for the trained model.

## Model

- Input: 30 chronological app tokens
- Vocabulary: 118 tokens, including an out-of-vocabulary token
- Architecture: embedding -> bidirectional LSTM -> two dense/dropout blocks -> softmax classifier
- Output: probability distribution over the next app

See [MODEL_SUMMARY.md](MODEL_SUMMARY.md) for the complete architecture, [best_hyperparams.json](best_hyperparams.json) for hyperparameters, and [index_to_app.json](index_to_app.json) for the token-to-app mapping.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Verify the included test predictions.
python evaluate.py

# Retrain using the prepared chronological splits.
python train.py --epochs 10
```

## Make a Prediction

Pass exactly 30 comma-separated app names, ordered from oldest to newest:

```bash
python predict.py "google,chrome,gmail,google,chrome,gmail,google,chrome,gmail,google,chrome,gmail,google,chrome,gmail,google,chrome,gmail,google,chrome,gmail,google,chrome,gmail,google,chrome,gmail,google,chrome,gmail"
```

The script prints the top five likely next apps and their probabilities.

## Files

| File | Purpose |
| --- | --- |
| `X_train.npy`, `X_val.npy`, `X_te.npy` | Tokenized 30-event input sequences |
| `y_train.npy`, `y_val.npy`, `y_te.npy` | Next-app labels for each split |
| `app_predictor_model (1).keras` | Trained Keras model |
| `tokenizer.pkl` | Keras tokenizer used for app names |
| `y_hat.npy` | Included test-set predictions |
| `train.py` | Reproduces the training workflow from the prepared data |
| `predict.py` | Runs top-k inference from app names |
| `evaluate.py` | Reports held-out top-1 accuracy |

## Viewing Binary Artifacts

`.npy`, `.keras`, and `.pkl` files are binary formats, so GitHub cannot render their raw contents as text. This repository includes [MODEL_SUMMARY.md](MODEL_SUMMARY.md) and JSON configuration files for browser-readable documentation. In VS Code, install the recommended extensions to inspect `.npy` and `.pkl` files directly.

## Data Source and Citation

The prepared sequences in this project were derived from [LSApp: Large dataset of Sequential mobile App usage](https://github.com/aliannejadi/LSApp). LSApp contains consented sequential app-usage events collected from 292 participants, including timestamps, app names, event types, session IDs, and anonymized user IDs. This project preprocesses those events into chronological 30-app windows for next-app prediction.

If you use LSApp or this derived dataset, please cite its associated publication:

```bibtex
@article{AliannejadiTOIS21,
  author  = {Mohammad Aliannejadi and Hamed Zamani and Fabio Crestani and W. Bruce Croft},
  title   = {Context-Aware Target Apps Selection and Recommendation for Enhancing Personal Mobile Assistants},
  journal = {ACM Transactions on Information Systems},
  year    = {2021}
}
```

## Data Note

The prepared arrays are derived app-usage behavior data. Confirm that you have permission to publish the source data and that it contains no personal or identifying information before making the repository public.
