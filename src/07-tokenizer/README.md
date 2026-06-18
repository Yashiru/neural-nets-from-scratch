# 07 | Tokenizer

`06` read text one character at a time. That keeps the vocabulary tiny (65 symbols for
Shakespeare) but makes every sequence as long as the text itself, and the model burns
capacity relearning that `t`, `h`, `e` spell a word. This step swaps the character
vocabulary for a learned **subword** one: byte-pair encoding (BPE), the tokenizer under
GPT-2 and most LLMs. The trade is deliberate, a bigger vocabulary in exchange for much
shorter sequences.

Everything here is from scratch: train merges on raw bytes, encode and decode with an
exact round-trip, then two experiments that measure what the tokenizer actually buys and
who it leaves behind.

## Byte-pair encoding: grow the vocabulary by merging

BPE starts from the 256 possible bytes and repeatedly does one thing: find the most
frequent adjacent pair of tokens and merge it into a single new token. Each merge adds one
entry to the vocabulary and shortens every place that pair occurred.

```python
def train(text, vocab_size, on_step=None):
    word_freqs = Counter(re.findall(GPT2_PAT, text))
    ids = {tuple(w.encode("utf-8")): f for w, f in word_freqs.items()}
    merges, vocab = {}, {i: bytes([i]) for i in range(256)}
    for k in range(vocab_size - 256):
        stats = get_stats(ids)              # count adjacent pairs
        pair = max(stats, key=stats.get)    # the most frequent one
        idx = 256 + k
        ids = merge_vocab(ids, pair, idx)   # replace it everywhere
        merges[pair] = idx                  # remember the recipe, in order
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
    return merges, vocab
```

Two dictionaries carry the result. `merges` maps a pair to the new token id, in the order
it was learned, and `vocab` maps every id back to its raw bytes. `merges` is what `encode`
replays; `vocab` is what `decode` looks up. Starting from bytes (not characters) means any
input is representable: no out-of-vocabulary symbol can ever appear, accents and emojis
included.

Watching the merges form on Shakespeare is the clearest explanation of what BPE learns.
The first merges are the most common English byte pairs, and they compose into suffixes,
whole words, and finally character names as the vocabulary grows:

```
256: ' t'    257: 'he'    258: ' a'    267: ' the'   288: ' you'   296: ' and'
...
720: 'GLOUCESTER'   1156: 'HENRY'   1741: 'NORTHUMBERLAND'
```

The vocabulary self-organizes from bytes up to letter pairs, to suffixes, to frequent
words, to names. The merges are frequency-driven and corpus-specific, which turns out to
be the whole point of experiment 2.

## Pre-tokenization: never merge across a word boundary

A naive BPE counts pairs over the entire byte stream. That is both slow and wrong. Slow
because a 1.1 MB corpus is a million positions to rescan on every merge; wrong because
nothing stops it merging a space with the next letter, or `". "` into one token, producing
tokens that straddle boundaries and generalize badly.

The fix, from GPT-2, is to split the text with a regex first, then run BPE strictly inside
each piece:

```python
GPT2_PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
```

Each alternative grabs one kind of run: contractions (`'s`, `'ll`), a run of letters with
an optional leading space (` word`), a run of digits, a run of punctuation, or whitespace.
The leading space is attached to the following word, so ` the` is one chunk and merges
never reach back across the gap. (`\p{L}` and `\p{N}` are Unicode letter and number
classes, which is why this needs the `regex` module, not the standard `re`.)

It also makes training fast. Instead of a million-long list, I count over the **unique**
chunks weighted by how often each appears (`Counter`), so `get_stats` adds `freq`, not `1`:

```python
def get_stats(ids):
    counts = {}
    for word, freq in ids.items():
        for pair in zip(word, word[1:]):
            counts[pair] = counts.get(pair, 0) + freq
    return counts
```

Shakespeare has a few tens of thousands of distinct chunks for a million characters, so
each merge scans the small set, not the full stream. The 4096-vocab training finishes in
under 40 seconds.

## Encoding: replay the merges, lowest rank first

Encoding has to reproduce exactly the order training merged things. The trap is to apply
merges by frequency or arbitrarily; the rule is to always apply the merge that was learned
**earliest** (lowest id) among the pairs currently present, and repeat until none apply.

```python
def encode_chunk(chunk, merges):
    ids = list(chunk)
    while len(ids) >= 2:
        pairs = set(zip(ids, ids[1:]))
        pair = min(pairs, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break                            # no learnable pair left
        ids = merge(ids, pair, merges[pair])
    return ids
```

`encode` just splits the text with the same regex and runs this per chunk. Decoding is the
easy direction: concatenate the raw bytes for each id and decode UTF-8, with
`errors="replace"` so an arbitrary token sequence from a model cannot crash the round-trip.
A sample with accents, an apostrophe and emojis survives the full loop:

```
C'était un grand plaisir de vous rencontrer! 😊✨ #Python #Tokenization
```

## Experiment 1: more vocabulary, fewer tokens

The first thing to measure is the trade itself. Train on Shakespeare (1,115,394 bytes) at
growing vocab sizes, encode the whole corpus, and look at how many tokens it takes.
`bytes/token` is the compression ratio: how many raw bytes each token now carries.

| vocab_size | merges | tokens | bytes/token |
|-----------:|-------:|----------:|------------:|
| 256  | 0     | 1,115,394 | 1.00 |
| 512  | 256   | 575,345   | 1.94 |
| 1024 | 768   | 459,760   | 2.43 |
| 2048 | 1,792 | 388,514   | 2.87 |
| 4096 | 3,840 | 344,095   | 3.24 |

At vocab 256 there are no merges, so every byte is its own token and compression is exactly
1.00: the character-level baseline `06` ran on. The first 256 merges nearly **halve** the
sequence (1.94 bytes/token), because the cheapest, most common pairs do the heavy lifting.
After that it is diminishing returns: from a 512-token vocabulary to 4096 (8x larger)
compression only climbs from 1.94 to 3.24, since every new merge targets a rarer pattern.

That curve is the whole tokenizer design decision in one table. A bigger vocabulary means
shorter sequences (cheaper attention, longer effective context) but a wider, more
expensive embedding table and softmax, and rarer tokens seen less often during training.
Real tokenizers sit where the curve has flattened: GPT-2 uses about 50k.

## Experiment 2: the non-English tax

BPE merges are learned from a corpus, so they bake in whichever language trained them. To
measure what that costs everyone else, I use the **same book** in two languages (Verne,
*Twenty Thousand Leagues under the Sea* / *Vingt mille lieues sous les mers*), train one
tokenizer per language at vocab 2500, and count tokens three ways.

| config | tokens | words | tok/word |
|:--|--:|--:|--:|
| EN text, EN tokenizer | 186,004 | 104,672 | 1.777 |
| FR text, EN tokenizer | 465,429 | 143,308 | 3.248 |
| FR text, FR tokenizer | 289,641 | 143,308 | 2.021 |

The headline: encoding the same book costs 186k tokens in English but **465k in French
under the English tokenizer, 2.5x as many**. For a model billed and context-limited per
token, French is two and a half times more expensive to read and write, for identical
content.

That 2.5x decomposes cleanly into two compounding effects:

- **Verbosity.** French uses more words for the same meaning: 143,308 vs 104,672, a factor
  of **1.37**. This is the language, not the tokenizer.
- **Fertility.** Each French word also costs more tokens under the English tokenizer:
  3.248 vs 1.777 tok/word, a factor of **1.83**. The English merges rarely fit French
  strings, so French falls back toward raw bytes (and accented characters are two bytes
  each in UTF-8, so they often start as two tokens).

And `1.37 x 1.83 = 2.5`, the total. The third row isolates the tokenizer's share of the
blame. Give French its own merges and fertility drops from 3.248 to 2.021 tok/word: the
English tokenizer is **1.61x heavier on French than a French one would be**. The leftover
gap (2.021 vs 1.777, about 1.14x) is intrinsic, French is a little denser even with a fair
tokenizer, but most of the tax is not intrinsic. It is the direct, measurable consequence
of training the merges on English.

> This is the from-scratch version of a real fairness problem. Production tokenizers are
> trained on English-dominated corpora, so speakers of under-represented languages pay more
> per word, get less usable context, and often see worse quality, all decided by a
> preprocessing step that ran before the model ever existed.

## Running the experiments

From the repo root:

```bash
python3 src/07-tokenizer/main.py
```

Both experiments run in turn (about 3 to 5 minutes total). `experiment_vocab_size` trains
on Shakespeare at five vocab sizes; `experiment_languages` trains an English and a French
tokenizer on the matched Verne corpora and counts tokens for the three configs above. The
corpora are gitignored; `data/README.md` has the one-line fetch commands. `main()` is the
original demo (train, print the vocabulary and merges, round-trip a sample with accents and
emojis) and is left commented in `__main__`.

## Takeaways

- **BPE is just repeated greedy merging.** Count adjacent pairs, merge the most frequent
  into a new token, record the recipe in order, repeat. Starting from the 256 bytes
  guarantees every input is representable.
- **Pre-tokenization is not optional.** Splitting on a regex first keeps merges inside word
  boundaries (quality) and lets training count over unique chunks instead of the full
  stream (speed).
- **Encoding must replay merges in learned order.** Always apply the lowest-id pair
  present; apply them in any other order and the round-trip breaks.
- **Vocabulary size buys compression with diminishing returns.** The first merges nearly
  halve the sequence; later ones chase rarer patterns. That curve is why real vocabularies
  land in the tens of thousands, not the millions.
- **The tokenizer is not language-neutral.** The same book costs 2.5x the tokens in French
  under an English tokenizer, and 1.6x of that is the tokenizer's training, not the
  language. Preprocessing choices have downstream consequences that look like model
  behavior but are decided before the model runs.
- A pointer to what is next: this tokenizer produces exactly what the `06` GPT could
  consume. Wiring it in, swapping the character vocabulary for these subword tokens, lets
  the model read ` the` in one step instead of three, with the shorter sequences
  experiment 1 measured.
