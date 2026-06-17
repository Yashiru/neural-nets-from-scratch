"""Experiment: 05-wavenet.

Next idea I want to try: grow the flat MLP into a hierarchical, tree-like
network (à la WaveNet) that fuses the context a few characters at a time
instead of all at once, and wrap the layers in a small reusable module API.
Not built yet.

Run from the repo root with:  python3 src/05-wavenet/main.py
"""

import torch
from common import display as ui
from torch.nn import functional as F
from common.data import load_words, build_vocab
from common.NeuralNetwork.basics import (
    Sequential,
    Linear,
    Tanh,
    BatchNorm1d,
    embedding,
    FlattenConsecutive
)

# ----------------------------- Hyperparameters ------------------------------ #
BLOCK_SIZE = 8  # how many characters of context feed each prediction
N_EMBD = 20  # embedding dimensions per character
N_HIDDEN = 200  # neurons in the hidden layer
BATCH_SIZE = 32  # examples per minibatch (this is `n` in the backward formulas)
STEPS = 50_000  # optimization steps (used when TRAIN is True)
LR = 0.07  # base lr
LR_FINE = LR / 10  # fine-tuning lr after the decay point
FLATTEN_CONSECUTIVE = 2  # how many consecutive characters to flatten into one vector


def main():

    _, itos, n_words, (Xtr, Ytr), (Xdev, Ydev), (Xte, Yte) = load_data()
    vocab_size = len(itos)

    model = Sequential(
        [
            embedding(vocab_size, N_EMBD),
            FlattenConsecutive(2),
            Linear(N_EMBD * 2, N_HIDDEN, bias=False),  # (B,4,40)→(B,4,200)
            BatchNorm1d(N_HIDDEN),
            Tanh(),
            FlattenConsecutive(2),
            Linear(N_HIDDEN * 2, N_HIDDEN, bias=False),  # (B,2,400)→(B,2,200)
            BatchNorm1d(N_HIDDEN),
            Tanh(),
            FlattenConsecutive(2),
            Linear(N_HIDDEN * 2, N_HIDDEN, bias=False),  # (B,1,400)→(B,200) squeeze
            BatchNorm1d(N_HIDDEN),
            Tanh(),
            Linear(N_HIDDEN, vocab_size),
        ]
    )

    ui.banner(
        "MANUAL BACKPROPAGATION  (tensor-level)",
        f"MLP + BatchNorm, gradients by hand   (context = {BLOCK_SIZE} chars)",
    )

    _, itos, n_words, (Xtr, Ytr), (Xdev, Ydev), (Xte, Yte) = load_data()
    vocab_size = len(itos)

    ui.section("Dataset")
    ui.kv("words", f"{n_words:,}")
    ui.kv("vocabulary", f"{vocab_size} tokens  (a-z + '.')")
    ui.kv("train examples", f"{Xtr.shape[0]:,}")

    ui.section("Model")
    ui.kv(
        "architecture",
        f"{BLOCK_SIZE}×{N_EMBD} → FlattenConsecutive({FLATTEN_CONSECUTIVE}) → Linear({N_EMBD * FLATTEN_CONSECUTIVE}, {N_HIDDEN}) → BatchNorm1d({N_HIDDEN}) → Tanh() → "
        f"FlattenConsecutive({FLATTEN_CONSECUTIVE}) → Linear({N_HIDDEN * FLATTEN_CONSECUTIVE}, {N_HIDDEN}) → BatchNorm1d({N_HIDDEN}) → Tanh() → "
        f"FlattenConsecutive({FLATTEN_CONSECUTIVE}) → Linear({N_HIDDEN * FLATTEN_CONSECUTIVE}, {N_HIDDEN}) → BatchNorm1d({N_HIDDEN}) → Tanh() → Linear({N_HIDDEN}, {vocab_size})"
    )
    ui.kv("parameters", f"{sum(p.nelement() for p in model.parameters()):,}")
    ui.kv("batch size (n)", BATCH_SIZE)

    # --------------------- Train with the manual gradients ---------------------- #
    ui.section("Training  (manual gradients only)")
    train(model, Xtr, Ytr)

    # ------------------------------ Generate names ------------------------------ #
    samples = [generate(model, itos, BLOCK_SIZE) for _ in range(20)]
    first = next((s for s in samples if s[0]), samples[0])

    ui.section("Sampling Trace  (first generated name)")
    ui.trace(first[1])

    ui.section(f"Generated Names  ({len(samples)} samples)")
    ui.name_list([s for s in samples if s[0]])


def train(model, Xtr, Ytr):
    """Train using ONLY the hand-derived gradients — no loss.backward()."""
    model.train()
    decay_at = int(0.6 * STEPS)
    losses = []
    with ui.training_progress(STEPS) as step_done:
        for step in range(STEPS):
            ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,))
            Xb, Yb = Xtr[ix], Ytr[ix]

            loss, c = model(Xb, Yb)
            model.backward()
            model.optimize(lr=LR_FINE if step >= decay_at else LR)

            if step % 100 == 0 or step == STEPS - 1:
                losses.append(loss.item())
            step_done(loss=loss.item(), lr=LR_FINE if step >= decay_at else LR)

    ui.kv("final loss", f"{losses[-1]:.4f}  nats  (last minibatch)")
    ui.kv("initial loss", f"{losses[0]:.4f}  nats")
    ui.loss_curve(losses, title="Training loss  (manual backprop)")


@torch.no_grad()
def generate(model, itos, block_size, max_len=40):
    model.eval()
    context = [0] * block_size
    out, steps = [], []
    while len(out) < max_len:
        _, logits = model(torch.tensor([context]))

        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        ctx = "".join(itos[i] for i in context)
        steps.append((ctx, itos[ix], float(probs[0, ix])))
        context = context[1:] + [ix]
        if ix == 0:  # '.' = end-of-name token
            break
        out.append(itos[ix])
    return "".join(out), steps


def load_data():
    """Load names, shuffle, and split 80/10/10 into train/dev/test tensors."""
    words = load_words()
    stoi, itos = build_vocab(words)
    perm = torch.randperm(len(words)).tolist()
    words = [words[i] for i in perm]
    n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
    tr = build_dataset(words[:n1], stoi, BLOCK_SIZE)
    dev = build_dataset(words[n1:n2], stoi, BLOCK_SIZE)
    te = build_dataset(words[n2:], stoi, BLOCK_SIZE)
    return stoi, itos, len(words), tr, dev, te


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


if __name__ == "__main__":
    main()
