"""Experiment: 02-mlp.

Character-level MLP language model (Bengio 2003 style): each character is
embedded, the last `BLOCK_SIZE` embeddings are concatenated and fed through a
single tanh hidden layer, then a linear layer produces the next-char logits.
Run from the repo root with:  python3 src/02-mlp/main.py
"""
import torch
import torch.nn.functional as F

from common import display as ui
from common.data import load_words, build_vocab

SEED = 2147483647

# ----------------------------- Hyperparameters ------------------------------ #
BLOCK_SIZE = 3       # how many characters of context feed each prediction
N_EMBD = 10          # embedding dimensions per character
N_HIDDEN = 200       # neurons in the hidden layer
STEPS = 50_000       # optimization steps
BATCH_SIZE = 32      # examples per minibatch


def build_dataset(words, stoi, block_size):
    """Turn words into (context -> next char) tensors X (N, block_size), Y (N,)."""
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


def forward(params, X):
    """Run the MLP forward pass for a batch of contexts X, returning logits."""
    C, W1, b1, W2, b2 = params
    emb = C[X]                                   # (N, block_size, n_embd)
    h = torch.tanh(emb.view(emb.shape[0], -1) @ W1 + b1)   # (N, n_hidden)
    return h @ W2 + b2                           # (N, vocab_size)


@torch.no_grad()
def split_loss(params, X, Y):
    """Full-split cross-entropy (no minibatch noise), for honest evaluation."""
    return F.cross_entropy(forward(params, X), Y).item()


@torch.no_grad()
def generate(params, itos, block_size, generator, max_len=40):
    """Sample one name autoregressively, returning (name, trace).

    The trace is a list of (context, next_char, prob) tuples, fed to ui.trace.
    """
    context = [0] * block_size
    out, steps = [], []
    while len(out) < max_len:
        logits = forward(params, torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=generator).item()
        ctx = "".join(itos[i] for i in context)
        steps.append((ctx, itos[ix], float(probs[0, ix])))
        context = context[1:] + [ix]
        if ix == 0:                              # '.' = end-of-name token
            break
        out.append(itos[ix])
    return "".join(out), steps


def main():
    g = torch.Generator().manual_seed(SEED)

    ui.banner("MULTI-LAYER PERCEPTRON LANGUAGE MODEL",
              f"embed → tanh({N_HIDDEN}) → logits   (context = {BLOCK_SIZE} chars)")

    # ------------------------------- Load + split ------------------------------- #
    words = load_words()
    stoi, itos = build_vocab(words)
    vocab_size = len(itos)

    perm = torch.randperm(len(words), generator=g).tolist()
    words = [words[i] for i in perm]
    n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
    Xtr, Ytr = build_dataset(words[:n1], stoi, BLOCK_SIZE)
    Xdev, Ydev = build_dataset(words[n1:n2], stoi, BLOCK_SIZE)
    Xte, Yte = build_dataset(words[n2:], stoi, BLOCK_SIZE)

    ui.section("Dataset")
    ui.kv("words", f"{len(words):,}")
    ui.kv("vocabulary", f"{vocab_size} tokens  (a-z + '.')")
    ui.kv("context length", f"{BLOCK_SIZE} chars")
    ui.kv("train examples", f"{Xtr.shape[0]:,}")
    ui.kv("val / test", f"{Xdev.shape[0]:,} / {Xte.shape[0]:,}")

    # ------------------------------ Build the model ----------------------------- #
    params = [
        torch.randn((vocab_size, N_EMBD), generator=g),
        torch.randn((BLOCK_SIZE * N_EMBD, N_HIDDEN), generator=g),
        torch.randn(N_HIDDEN, generator=g),
        torch.randn((N_HIDDEN, vocab_size), generator=g),
        torch.randn(vocab_size, generator=g),
    ]
    for p in params:
        p.requires_grad = True
    n_params = sum(p.nelement() for p in params)

    ui.section("Model")
    ui.kv("architecture", f"{BLOCK_SIZE}×{N_EMBD} → tanh({N_HIDDEN}) → {vocab_size}")
    ui.kv("embedding dim", N_EMBD)
    ui.kv("hidden units", N_HIDDEN)
    ui.kv("parameters", f"{n_params:,}")

    # ------------------------------ Train the model ----------------------------- #
    ui.section("Training")
    ui.kv("steps", f"{STEPS:,}")
    ui.kv("batch size", BATCH_SIZE)
    ui.kv("learning rate", "0.1 → 0.01  (decay at 60%)")
    ui.console.print()

    losses = []
    decay_at = int(0.6 * STEPS)
    with ui.training_progress(STEPS) as step_done:
        for step in range(STEPS):
            ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,), generator=g)
            loss = F.cross_entropy(forward(params, Xtr[ix]), Ytr[ix])

            for p in params:
                p.grad = None
            loss.backward()

            lr = 0.1 if step < decay_at else 0.01
            for p in params:
                p.data += -lr * p.grad

            if step % 100 == 0 or step == STEPS - 1:
                losses.append(loss.item())
            step_done(loss=loss.item(), lr=lr)

    ui.kv("initial loss", f"{losses[0]:.4f}  nats")
    ui.kv("final loss", f"{losses[-1]:.4f}  nats  (last minibatch)")
    ui.loss_curve(losses, title="Training loss  (minibatch, sampled every 100 steps)")

    # -------------------------------- Evaluation -------------------------------- #
    ui.section("Evaluation  (full-split cross-entropy)")
    ui.kv("train loss", f"{split_loss(params, Xtr, Ytr):.4f}  nats")
    ui.kv("val loss", f"{split_loss(params, Xdev, Ydev):.4f}  nats")
    ui.kv("test loss", f"{split_loss(params, Xte, Yte):.4f}  nats")

    # ------------------------------ Generate names ------------------------------ #
    samples = [generate(params, itos, BLOCK_SIZE, g) for _ in range(20)]
    first = next((s for s in samples if s[0]), samples[0])

    ui.section("Sampling Trace  (first generated name)")
    ui.trace(first[1])

    ui.section(f"Generated Names  ({len(samples)} samples · seed {SEED})")
    ui.name_list([s for s in samples if s[0]])


if __name__ == "__main__":
    main()
