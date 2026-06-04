"""Experiment: 01-bigram.

Scratch space for this experiment. Replace this with your implementation.
Run from the repo root with:  python3 src/01-bigram/main.py
"""
import math

import torch
import torch.nn.functional as F

from common import display as ui
from common.data import load_words, build_vocab

SEED = 42

def main():
    bigram()
    nn_bigram()

def bigram():
    # load the list of words in the file data/names.txt
    mots = load_words()
    stoi, itos = build_vocab(mots)

    # create a bigram matrix N of shape (27, 27) to count the number of occurrences of each bigram
    N = torch.zeros((27, 27), dtype=torch.int32)
    for mot in mots:
        chs = ['.'] + list(mot) + ['.']         # Add start and end tokens
        for ch1, ch2 in zip(chs, chs[1:]):      # zip(chs, chs[1:]) creates pairs of consecutive characters
            i = stoi[ch1]
            j = stoi[ch2]
            N[i, j] += 1

    # convert the counts to probabilities by normalizing each row of N to sum to 1, call the resulting matrix P
    P = (N + 1).float()      # +1 partout : plus aucune proba nulle
    P = P / P.sum(1, keepdim=True)

    # sample a handful of names from the model (deterministic via the seed)
    g = torch.Generator().manual_seed(SEED)
    n_samples = 8
    samples = [ui.sample_name(P, itos, g) for _ in range(n_samples)]

    # Evaluate quality
    log_likelihood = 0.0
    n = 0
    for mot in mots:
        chs = ['.'] + list(mot) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            prob = P[stoi[ch1], stoi[ch2]]
            log_likelihood += torch.log(prob)
            n += 1
    nll = float(-log_likelihood / n)

    # print a pro and beautiful detailed result of the whole process: the bigram
    # matrix, the probabilities, the model quality, and the generated names
    ui.banner("BIGRAM CHARACTER-LEVEL LANGUAGE MODEL",
              "count -> smooth -> normalize -> sample")

    ui.section("Dataset")
    ui.kv("words loaded", f"{len(mots):,}")
    ui.kv("vocabulary", f"{N.shape[0]} tokens  (a-z + '.')")
    ui.kv("total bigrams", f"{int(N.sum()):,}")
    longest = max(samples, key=lambda s: len(s[0]))[0] if samples else ""
    ui.kv("longest sample", f"{len(longest)} chars")

    ui.section("Bigram Count Matrix  N[i, j] = count( i -> j )")
    ui.matrix(N, itos)

    ui.section("Bigram Probabilities  P[i, j] = (N + 1) / sum_j (N + 1)")
    ui.distribution(P, itos, stoi["."])

    ui.section("Model Quality")
    ui.kv("avg neg log-lik", f"{nll:.4f}  nats/bigram")
    ui.kv("perplexity", f"{math.exp(nll):.2f}")
    ui.kv("scored bigrams", f"{n:,}")

    ui.section("Sampling Trace  (first generated name)")
    ui.trace(samples[0][1])

    ui.section(f"Generated Names  ({n_samples} samples - seed 41)")
    ui.name_list(samples)

def nn_bigram():
    """Same as bigram, but using a simple neural network instead of a bigram matrix."""

    # load the list of words in the file data/names.txt
    mots = load_words()
    stoi, itos = build_vocab(mots)
    xs, ys = [], []
    for mot in mots:
        chs = ['.'] + list(mot) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)

    # Create a weight matrix W of shape (27, 27) to represent the bigram probabilities, and enable gradient tracking on W
    W = torch.randn((27, 27), requires_grad=True)

    # Train the model for 300 steps using the negative log-likelihood loss and stochastic gradient descent.
    losses = []
    for step in range(300):
        # forward
        xenc = F.one_hot(xs, num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)
        loss = -probs[torch.arange(len(ys)), ys].log().mean()   # NLL
        losses.append(loss.item())

        # backward
        W.grad = None
        loss.backward()

        # update
        W.data += -75 * W.grad # gradient descent with learning rate 75 (we use a high learning rate because the model is very simple and we want to converge fast)

    # the learned row-stochastic transition matrix = softmax over each row of W
    with torch.no_grad():
        P = torch.softmax(W, dim=1)

    g = torch.Generator().manual_seed(SEED)
    n_samples = 8
    samples = [ui.sample_name(P, itos, g) for _ in range(n_samples)]

    # reuse the exact same display helpers as the counting model above
    ui.banner("NEURAL BIGRAM  (single linear layer)",
              "one-hot -> W -> softmax -> sample")

    ui.section("Training")
    ui.kv("steps", str(len(losses)))
    ui.kv("initial loss", f"{losses[0]:.4f}  nats/bigram")
    ui.kv("final loss", f"{losses[-1]:.4f}  nats/bigram")
    ui.loss_curve(losses)

    ui.section("Learned Probabilities  P[i, j] = softmax(W)[i, j]")
    ui.distribution(P, itos, stoi["."])

    ui.section("Sampling Trace  (first generated name)")
    ui.trace(samples[0][1])

    ui.section(f"Generated Names  ({n_samples} samples - seed {SEED})")
    ui.name_list(samples)


if __name__ == "__main__":
    main()
