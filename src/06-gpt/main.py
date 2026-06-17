"""Experiment: 06-gpt.

Next idea I want to try: build a small decoder-only Transformer from scratch
(token + positional embeddings, masked self-attention, residual MLP blocks)
and see how self-attention compares to the fixed-window MLP on the same names.
Not built yet.

Run from the repo root with:  python3 src/06-gpt/main.py
"""

import torch
from common import display as ui
from torch.nn import functional as F
from common.data import load_words, build_vocab
from common.NeuralNetwork.GPT import GPT

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
    # Ytr contains the next character for each context in Xtr, so it has shape (N,).
    # Xtr contains the context for each next character, so it has shape (N, BLOCK_SIZE).
    vocab_size = len(itos)

    model = GPT(vocab_size, N_EMBD, n_head=4, n_layer=4)

    ui.banner(
        "MANUAL BACKPROPAGATION  (tensor-level)",
        f"MLP + BatchNorm, gradients by hand   (context = {BLOCK_SIZE} chars)",
    )

    ui.section("Dataset")
    ui.kv("words", f"{n_words:,}")
    ui.kv("vocabulary", f"{vocab_size} tokens  (a-z + '.')")
    ui.kv("train examples", f"{Xtr.shape[0]:,}")

    ui.section("Model")
    ui.kv(
        "architecture",
        f"GPT(vocab_size={vocab_size}, n_embd={N_EMBD}, n_head=4, n_layer=4)",
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
    decay_at = int(0.6 * STEPS)
    losses = []
    with ui.training_progress(STEPS) as step_done:
        for step in range(STEPS):
            ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,))
            Xb, Yb = Xtr[ix], Ytr[ix]
            logits = model(Xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Yb.view(-1))

            model.zero_grad()
            loss.backward()
            model.optimize(lr=LR_FINE if step >= decay_at else LR)

            if step % 100 == 0 or step == STEPS - 1:
                losses.append(loss.item())
            step_done(loss=loss.item(), lr=LR_FINE if step >= decay_at else LR)

    ui.kv("final loss", f"{losses[-1]:.4f}  nats  (last minibatch)")
    ui.kv("initial loss", f"{losses[0]:.4f}  nats")
    ui.loss_curve(losses, title="Training loss  (manual backprop)")


@torch.no_grad()
def generate(model, itos, block_size, max_len=40):
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
            context = context[1:] + [ix]
            Y.append(context)
    return torch.tensor(X), torch.tensor(Y)


if __name__ == "__main__":
    main()
