# neural-nets-from-scratch

My personal playground for building neural networks from scratch, one idea at a
time, to understand how they actually work end to end. I treat each experiment
as a self-contained build, from a simple character-level language model up to a
small Transformer and a tokenizer.

## Experiments

| # | Folder | Focus |
| --- | --- | --- |
| 01 | [`src/01-bigram`](src/01-bigram/README.md) | Character-level bigram model: counting and the neural-net view |
| 02 | [`src/02-mlp`](src/02-mlp/README.md) | MLP language model: embeddings, train/dev/test, tuning |
| 03 | [`src/03-activations-batchnorm`](src/03-activations-batchnorm/README.md) | Activation/gradient health, initialization, batch norm |
| 04 | [`src/04-manual-backprop`](src/04-manual-backprop/README.md) | Backpropagation by hand at the tensor level |
| 05 | [`src/05-wavenet`](src/05-wavenet/README.md) | Hierarchical (tree-like) network and a small module API |
| 06 | [`src/06-gpt`](src/06-gpt/README.md) | A decoder-only Transformer (self-attention) from scratch |
| 07 | [`src/07-tokenizer`](src/07-tokenizer/README.md) | Byte-Pair Encoding tokenizer from scratch |

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
