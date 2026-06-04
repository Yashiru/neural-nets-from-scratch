# neural-nets-from-scratch

A personal playground for building neural networks from scratch, one idea at a
time, to understand how they actually work end to end. Each experiment is a
self-contained step, from a simple character-level language model up to a small
Transformer and a tokenizer.

## Experiments

| # | Folder | Focus |
| --- | --- | --- |
| 01 | `src/01-bigram` | Character-level bigram model: counting and the neural-net view |
| 02 | `src/02-mlp` | MLP language model: embeddings, train/dev/test, tuning |
| 03 | `src/03-activations-batchnorm` | Activation/gradient health, initialization, batch norm |
| 04 | `src/04-manual-backprop` | Backpropagation by hand at the tensor level |
| 05 | `src/05-wavenet` | Hierarchical (tree-like) network and a small module API |
| 06 | `src/06-gpt` | A decoder-only Transformer (self-attention) from scratch |
| 07 | `src/07-tokenizer` | Byte-Pair Encoding tokenizer from scratch |

## Setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

## Running an experiment

```bash
python3 src/01-bigram/main.py
```

Datasets go in `data/` (gitignored). Shared helpers live in `common/`.
