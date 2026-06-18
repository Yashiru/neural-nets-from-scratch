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

- `verne-en.txt` / `verne-fr.txt`: the *same* book (Jules Verne, *Twenty Thousand
  Leagues under the Sea* / *Vingt mille lieues sous les mers*) in English (~0.6 MB)
  and French (~0.9 MB), from Project Gutenberg. A matched parallel pair used by the
  `07-tokenizer` experiment to compare token counts across languages on identical
  content (the non-English disadvantage). The English file also doubles as a smaller
  English corpus. Both fetch commands strip the Gutenberg header/license boilerplate:

  ```sh
  curl -sSL "https://www.gutenberg.org/cache/epub/164/pg164.txt" \
    | awk '/\*\*\* *START OF/{f=1;next} /\*\*\* *END OF/{f=0} f' > data/verne-en.txt
  curl -sSL "https://www.gutenberg.org/cache/epub/5097/pg5097.txt" \
    | awk '/\*\*\* *START OF/{f=1;next} /\*\*\* *END OF/{f=0} f' > data/verne-fr.txt
  ```
