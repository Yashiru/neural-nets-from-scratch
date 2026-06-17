# Data

Drop datasets here (e.g. `names.txt`, a text corpus for the GPT experiment).
Data files are gitignored by default; keep this directory in git via this README.

## Datasets

- `names.txt` — newline-separated first names (used by the MLP/WaveNet/GPT name
  experiments).
- `tinyshakespeare.txt` — ~1.1 MB of Shakespeare, the character-level corpus for
  the GPT experiment (`DATASET = "shakespeare"` in `src/06-gpt/main.py`). Fetch it
  with:

  ```sh
  curl -sSL -o data/tinyshakespeare.txt \
    https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
  ```
