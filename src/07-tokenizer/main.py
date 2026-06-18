"""Experiment: 07-tokenizer.

A Byte-Pair Encoding tokenizer from scratch (train merges on raw bytes,
encode/decode round-trip) to move past the character level and feed subword
units into the models above.

Optimization: GPT-2 style pre-tokenization. We first split the text with a
regex, then learn BPE *inside* each chunk. Pairs are counted over the unique
words weighted by their frequency (a few thousand types instead of a million
positions), and merges never cross a word boundary.

Run from the repo root with:  python3 src/07-tokenizer/main.py
"""

from common.data import load_text
from common import display as ui
import regex as re
from collections import Counter

GPT2_PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def train(text, vocab_size, on_step=None):
    """Train BPE merges on `text` up to `vocab_size`, returning (merges, vocab).

    Pure training core, reused by `main` and by the experiments. Pass `on_step`
    to receive (step, pair, count, vocab_size) callbacks for a progress bar.
    """
    word_freqs = Counter(re.findall(GPT2_PAT, text))
    ids = {tuple(w.encode("utf-8")): f for w, f in word_freqs.items()}
    merges, vocab = {}, {i: bytes([i]) for i in range(256)}
    for k in range(vocab_size - 256):
        stats = get_stats(ids)
        if not stats:
            break  # no pair left to merge
        pair = max(stats, key=stats.get)  # the most frequent pair
        idx = 256 + k
        ids = merge_vocab(ids, pair, idx)
        merges[pair] = idx
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        if on_step:
            on_step(k + 1, pair, stats[pair], len(vocab))
    return merges, vocab


def main():
    vocab_size = 2500
    text_data = load_text("tinyshakespeare.txt")
    with ui.tokenizer_training_progress(vocab_size - 256) as step_done:
        merges, vocab = train(text_data, vocab_size, on_step=step_done)
    # Print the final vocabulary and merges
    print("Final Vocabulary:")
    for idx, token in vocab.items():
        print(f"{idx}: {token}")
    print("\nFinal Merges:")
    for pair, idx in merges.items():
        print(f"{pair}: {idx}")
        
    # encode and decode a complex sample text with accents, punctuation, special characters, emojis, etc...
    sample_text = "C'était un grand plaisir de vous rencontrer! 😊✨ #Python #Tokenization"
    encoded_ids = encode(sample_text, merges)
    decoded_text = decode(encoded_ids, vocab)
    print(f"\nSample Text: {sample_text}")
    print(f"Encoded IDs: {encoded_ids}")
    print(f"Decoded Text: {decoded_text}")
    assert sample_text == decoded_text, "Decoded text does not match the original text!"

def encode(text, merges):
    """Encode a string into token IDs, chunk by chunk (same split as in training)."""
    out = []
    for chunk in re.findall(GPT2_PAT, text):
        out.extend(encode_chunk(chunk.encode("utf-8"), merges))
    return out

def encode_chunk(chunk, merges):
    """Apply BPE to the bytes of a single chunk.

    On each pass we merge the present pair whose merge was learned earliest
    (lowest idx), until no known pair remains.
    """
    ids = list(chunk)
    while len(ids) >= 2:
        pairs = set(zip(ids, ids[1:]))
        # present pair with the lowest merge rank (inf = never learned)
        pair = min(pairs, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break  # none of the remaining pairs has a learned merge
        ids = merge(ids, pair, merges[pair])
    return ids

def decode(ids, vocab):
    """Decode a list of token IDs back into a string using the provided vocabulary."""
    tokens = [vocab[i] for i in ids]
    # errors="replace": an arbitrary byte sequence (generated text) can land on
    # incomplete UTF-8, so we replace instead of crashing.
    return b"".join(tokens).decode("utf-8", errors="replace")

def get_stats(ids):
    counts = {}
    for word, freq in ids.items():
        for pair in zip(word, word[1:]):
            counts[pair] = counts.get(pair, 0) + freq   # add freq, not 1
    return counts

def merge_vocab(ids, pair, idx):
    new_ids = {}
    for word, freq in ids.items():
        new_ids[merge(word, pair, idx)] = freq   # merge() returns a tuple, usable as a dict key
    return new_ids

def merge(ids, pair, idx):
    new_ids, i = tuple(), 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            new_ids += (idx,)  # replace the pair with the new token
            i += 2
        else:
            new_ids += (ids[i],)
            i += 1
    return new_ids


def count_tokens(text, merges):
    """Total tokens to encode `text`, encoding each unique chunk only once."""
    total = 0
    for chunk, freq in Counter(re.findall(GPT2_PAT, text)).items():
        total += len(encode_chunk(chunk.encode("utf-8"), merges)) * freq
    return total


def experiment_vocab_size(filename="tinyshakespeare.txt",
                          vocab_sizes=(256, 512, 1024, 2048, 4096)):
    """Experiment 1: how the token count shrinks as the vocabulary grows.

    Train BPE on one corpus at several vocab sizes, encode the whole corpus, and
    report the token count and the compression ratio (bytes per token). More
    merges build longer tokens, so the same text needs fewer of them.
    """
    text = load_text(filename)
    n_bytes = len(text.encode("utf-8"))
    ui.section(f"Experiment 1: tokens vs vocabulary size ({filename})")
    ui.kv("corpus bytes", f"{n_bytes:,}")
    print(f"\n  {'vocab_size':>10} {'merges':>7} {'tokens':>10} {'bytes/token':>12}")
    for vs in vocab_sizes:
        with ui.tokenizer_training_progress(vs - 256,
                                            description=f"train vocab {vs}") as step:
            merges, _ = train(text, vs, on_step=step)
        n_tokens = count_tokens(text, merges)
        print(f"  {vs:>10} {len(merges):>7} {n_tokens:>10,} {n_bytes / n_tokens:>12.2f}")


def experiment_languages(en_file="verne-en.txt", fr_file="verne-fr.txt",
                         vocab_size=2500):
    """Experiment 2: the tokenization penalty for a non-English language.

    Same book in both languages. We train one tokenizer per language, then count
    tokens for three setups: English under its own tokenizer (baseline), French
    under the *English* tokenizer (the penalty), and French under a French
    tokenizer (the control). The gap between the last two shows the penalty comes
    from training the merges on English, not from French being harder.
    """
    en_text, fr_text = load_text(en_file), load_text(fr_file)
    ui.section(f"Experiment 2: English vs French, same book (vocab {vocab_size})")
    with ui.tokenizer_training_progress(vocab_size - 256, description="train EN") as s:
        en_merges, _ = train(en_text, vocab_size, on_step=s)
    with ui.tokenizer_training_progress(vocab_size - 256, description="train FR") as s:
        fr_merges, _ = train(fr_text, vocab_size, on_step=s)

    configs = [
        ("EN text / EN tok", en_text, en_merges),
        ("FR text / EN tok", fr_text, en_merges),
        ("FR text / FR tok", fr_text, fr_merges),
    ]
    print(f"\n  {'config':>16} {'tokens':>10} {'words':>8} {'tok/word':>9}")
    fertility = {}
    for label, text, merges in configs:
        n_tokens, n_words = count_tokens(text, merges), len(text.split())
        fertility[label] = n_tokens / n_words
        print(f"  {label:>16} {n_tokens:>10,} {n_words:>8,} {n_tokens / n_words:>9.3f}")
    penalty = fertility["FR text / EN tok"] / fertility["EN text / EN tok"]
    control = fertility["FR text / EN tok"] / fertility["FR text / FR tok"]
    print(f"\n  French under the English tokenizer: {penalty:.2f}x the tok/word of English.")
    print(f"  A French tokenizer erases most of it: {control:.2f}x heavier under the EN tokenizer.")


if __name__ == "__main__":
    # main()  # demo: train on Shakespeare, print the vocab, round-trip a sample
    experiment_vocab_size()
    experiment_languages()
