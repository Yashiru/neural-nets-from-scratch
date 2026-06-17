"""Experiment: 03-activations-batchnorm.

Same character-level MLP as 02-mlp (Bengio 2003 style): each character is
embedded, the last `BLOCK_SIZE` embeddings are concatenated and fed through a
single tanh hidden layer, then a linear layer produces the next-char logits.

Here I shift focus from architecture to the health of the signal. I scale the
output layer down so the first logits are near-uniform, which kills the loss
spike at step 0 and stops the early steps from being wasted just squashing
oversized weights.

Run from the repo root with:
    python3 src/03-activations-batchnorm/main.py    # train + evaluate + sample
"""

import sys
import matplotlib.pyplot as plt
import math

import torch
import torch.nn.functional as F

from common import display as ui
from common.data import load_words, build_vocab

SEED = 2147483647

# ----------------------------- Hyperparameters ------------------------------ #
BLOCK_SIZE = 3  # how many characters of context feed each prediction
N_EMBD = 20  # embedding dimensions per character
N_HIDDEN = 200  # neurons in the hidden layer
STEPS = 50000  # optimization steps
BATCH_SIZE = 32  # examples per minibatch
LR = 0.07  # base lr ≈ 10**-1.15 (from exp 02's lr probe); used until the decay point
LR_FINE = LR / 10  # fine-tuning lr, used after the decay point


def build_dataset(words, stoi, block_size):
    """Turn words into (context -> next char) tensors X (N, block_size), Y (N,)."""
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


def load_data(g):
    """Load names, shuffle, and split 80/10/10 into train/dev/test tensors."""
    words = load_words()
    stoi, itos = build_vocab(words)
    perm = torch.randperm(len(words), generator=g).tolist()
    words = [words[i] for i in perm]
    n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
    tr = build_dataset(words[:n1], stoi, BLOCK_SIZE)
    dev = build_dataset(words[n1:n2], stoi, BLOCK_SIZE)
    te = build_dataset(words[n2:], stoi, BLOCK_SIZE)
    return stoi, itos, len(words), tr, dev, te


def init_params(vocab_size, g):
    """Initialize the MLP parameters, scaling the output layer to tame the start.

    Plain `randn` weights make the first logits large and random, so the model
    starts wildly over-confident and the initial loss sits far above the uniform
    baseline (ln(vocab_size) ≈ 3.30 nats). Shrinking W2 and zeroing b2 keeps the
    first logits near zero, so training starts from a sane loss instead of
    burning the early steps just to undo a bad initialization.
    """
    params = [
        torch.randn((vocab_size, N_EMBD), generator=g),
        torch.randn((BLOCK_SIZE * N_EMBD, N_HIDDEN), generator=g)
        * (5 / 3)
        / ((BLOCK_SIZE * N_EMBD) ** 0.5),
        torch.ones(N_HIDDEN),
        torch.zeros(N_HIDDEN),
        torch.randn((N_HIDDEN, vocab_size), generator=g)
        * 0.01,  # shrink W2 so the first logits are ~0 (near-uniform softmax, no loss spike)
        torch.randn(vocab_size, generator=g)
        * 0,  # zero the output bias so nothing tilts the initial logits
    ]
    for p in params:
        p.requires_grad = True

    return params


def forward(params, X, bn, training=True):
    """Run the MLP forward pass for a batch of contexts X, returning (logits, h).

    BatchNorm normalizes the hidden pre-activations. While training we use the
    batch's own mean/std and fold them into the running estimates kept in `bn`;
    at eval time we reuse those estimates instead, so a single example (with no
    batch to compute stats from) still gets a sane, deterministic normalization.
    """
    C, W1, bngain, bnbias, W2, b2 = params
    emb = C[X]  # (N, block_size, n_embd)
    
    # pre-activation of the hidden layer
    hpreact = emb.view(emb.shape[0], -1) @ W1
    if training:
        bnmean = hpreact.mean(0, keepdim=True)
        bnstd  = hpreact.std(0, keepdim=True)
        with torch.no_grad():  # buffers, not parameters: update them off-graph
            bn["mean"] = 0.999 * bn["mean"] + 0.001 * bnmean
            bn["std"] = 0.999 * bn["std"] + 0.001 * bnstd
    else:
        bnmean, bnstd = bn["mean"], bn["std"]  # eval: reuse the running estimates
    hpreact = bngain * (hpreact - bnmean) / (bnstd + 1e-5) + bnbias

    h = torch.tanh(hpreact)

    return h @ W2 + b2, h


@torch.no_grad()
def split_loss(params, X, Y, bn):
    """Full-split cross-entropy (no minibatch noise), for honest evaluation."""
    logits, _ = forward(params, X, bn, training=False)
    return F.cross_entropy(logits, Y).item()


@torch.no_grad()
def generate(params, itos, block_size, generator, bn, max_len=40):
    """Sample one name autoregressively, returning (name, trace).

    The trace is a list of (context, next_char, prob) tuples, fed to ui.trace.
    """
    context = [0] * block_size
    out, steps = [], []
    while len(out) < max_len:
        logits, _ = forward(params, torch.tensor([context]), bn, training=False)
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=generator).item()
        ctx = "".join(itos[i] for i in context)
        steps.append((ctx, itos[ix], float(probs[0, ix])))
        context = context[1:] + [ix]
        if ix == 0:  # '.' = end-of-name token
            break
        out.append(itos[ix])
    return "".join(out), steps


@torch.no_grad()
def diagnose_init(params, Xtr, bn):
    """Run a forward pass with the initial parameters and display h histograms in the UI."""
    _, h = forward(params, Xtr, bn, training=True)  # batch stats over the full train split

    ui.histogram(
        h.view(-1)[::400].tolist(),
        bins=30,
        title="Initial hidden activations (tanh): spikes at ±1 = saturation",
        xlabel="h",
    )

    # Compuite saturation and dead neuron percentages
    total_neurons = h.shape[1]  # number of hidden neurons
    saturated = (h.abs() > 0.99).float().mean()
    dead = (h.abs() > 0.99).all(dim=0).sum()
    ui.kv("saturated neurons", f"{saturated:.2%}  (|h| > 0.99)")
    ui.kv("dead neurons", f"{dead} / {total_neurons}  ({dead / total_neurons:.2%})")


def main():
    g = torch.Generator().manual_seed(SEED)

    ui.banner(
        "MULTI-LAYER PERCEPTRON LANGUAGE MODEL",
        f"embed → tanh({N_HIDDEN}) → logits   (context = {BLOCK_SIZE} chars)",
    )

    # ------------------------------- Load + split ------------------------------- #
    _, itos, n_words, (Xtr, Ytr), (Xdev, Ydev), (Xte, Yte) = load_data(g)
    vocab_size = len(itos)

    ui.section("Dataset")
    ui.kv("words", f"{n_words:,}")
    ui.kv("vocabulary", f"{vocab_size} tokens  (a-z + '.')")
    ui.kv("context length", f"{BLOCK_SIZE} chars")
    ui.kv("train examples", f"{Xtr.shape[0]:,}")
    ui.kv("val / test", f"{Xdev.shape[0]:,} / {Xte.shape[0]:,}")

    # ------------------------------ Build the model ----------------------------- #
    params = init_params(vocab_size, g)
    n_params = sum(p.nelement() for p in params)

    # BatchNorm running stats: not learned. Updated by a moving average during
    # training, then reused (instead of batch stats) at eval / generation time.
    bn = {"mean": torch.zeros((1, N_HIDDEN)), "std": torch.ones((1, N_HIDDEN))}

    ui.section("Model")
    ui.kv("architecture", f"{BLOCK_SIZE}×{N_EMBD} → tanh({N_HIDDEN}) → {vocab_size}")
    ui.kv("embedding dim", N_EMBD)
    ui.kv("hidden units", N_HIDDEN)
    ui.kv("parameters", f"{n_params:,}")

    # ------------------------ Diagnose the initialization ----------------------- #
    ui.section("Initialization Diagnostics")
    diagnose_init(params, Xtr, bn)

    # ------------------------------ Train the model ----------------------------- #
    # base lr from exp 02's lr probe (≈10**-1.15), decayed tenfold after 60%.
    ui.section("Training")
    ui.kv("steps", f"{STEPS:,}")
    ui.kv("batch size", BATCH_SIZE)
    ui.kv("learning rate", f"{LR} → {LR_FINE}  (decay at 60%)")
    ui.console.print()

    losses = []
    decay_at = int(0.6 * STEPS)
    with ui.training_progress(STEPS) as step_done:
        for step in range(STEPS):
            ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,), generator=g)
            logits, _ = forward(params, Xtr[ix], bn, training=True)
            loss = F.cross_entropy(logits, Ytr[ix])

            for p in params:
                p.grad = None
            loss.backward()

            lr = LR if step < decay_at else LR_FINE
            for p in params:
                p.data += -lr * p.grad

            # comapre math.log(27) to the very first loss value to see how close we are to the uniform baseline at the start
            if step == 0:
                # display the comparison between loss.item():.4f and math.log(27) in the UI
                ui.kv(
                    "initial loss vs uniform baseline",
                    f"{loss.item():.4f} vs {math.log(vocab_size):.4f}  nats",
                )

            if step % 100 == 0 or step == STEPS - 1:
                losses.append(loss.item())
            step_done(loss=loss.item(), lr=lr)

    ui.kv("initial loss", f"{losses[0]:.4f}  nats")
    ui.kv("final loss", f"{losses[-1]:.4f}  nats  (last minibatch)")
    ui.loss_curve(losses, title="Training loss  (minibatch, sampled every 100 steps)")

    # -------------------------------- Evaluation -------------------------------- #
    ui.section("Evaluation  (full-split cross-entropy)")
    ui.kv("train loss", f"{split_loss(params, Xtr, Ytr, bn):.4f}  nats")
    ui.kv("val loss", f"{split_loss(params, Xdev, Ydev, bn):.4f}  nats")
    ui.kv("test loss", f"{split_loss(params, Xte, Yte, bn):.4f}  nats")

    # ------------------------------ Generate names ------------------------------ #
    samples = [generate(params, itos, BLOCK_SIZE, g, bn) for _ in range(20)]
    first = next((s for s in samples if s[0]), samples[0])

    ui.section("Sampling Trace  (first generated name)")
    ui.trace(first[1])

    ui.section(f"Generated Names  ({len(samples)} samples · seed {SEED})")
    ui.name_list([s for s in samples if s[0]])


if __name__ == "__main__":
    main()
