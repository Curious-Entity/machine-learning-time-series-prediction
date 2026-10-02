# Saved Model Summary

This file documents `app_predictor_model (1).keras`, which is a binary Keras model bundle.

| Property | Value |
| --- | --- |
| Input shape | `(None, 30)` |
| Output shape | `(None, 118)` |
| Trainable parameters | `900,982` |
| Model size | `3.44 MB` |

## Architecture

| Layer | Output shape | Parameters |
| --- | --- | ---: |
| InputLayer | `(None, 30)` | 0 |
| Embedding | `(None, 30, 128)` | 15,104 |
| Bidirectional LSTM | `(None, 512)` | 788,480 |
| Dense | `(None, 128)` | 65,664 |
| LeakyReLU | `(None, 128)` | 0 |
| Dropout | `(None, 128)` | 0 |
| Dense | `(None, 128)` | 16,512 |
| LeakyReLU | `(None, 128)` | 0 |
| Dropout | `(None, 128)` | 0 |
| Dense | `(None, 118)` | 15,222 |

The matching training configuration is in `best_hyperparams.json`; the input vocabulary is in `index_to_app.json`.

To regenerate the live model summary locally:

```bash
./.venv/bin/python view.py "app_predictor_model (1).keras"
```
