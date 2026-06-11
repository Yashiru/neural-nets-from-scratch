"""Experiment: 04-manual-backprop.

I take the MLP + BatchNorm from experiment 03, *delete loss.backward()*, and
reimplement the backward pass by hand, tensor by tensor, validating every
gradient against PyTorch's autograd. It's the tensor-level counterpart of the
scalar autograd engine I built earlier: same chain rule, but the new difficulty
is shapes and broadcasting (broadcast in forward  ==>  sum on that axis in
backward).

The run has three stages:
  - derive each manual gradient, simplest to hardest;
  - validate every gradient against autograd with cmp();
  - train using only the hand-derived gradients (set TRAIN = True).

The forward pass is written out op-by-op so every intermediate tensor has a name
I can backprop through and compare. Two backward steps use a collapsed analytic
form instead of exploding every micro-op:
  - cross_entropy collapsed to  dlogits = (softmax(logits) - onehot(Y)) / n
  - BatchNorm collapsed to the compact dhprebn formula.

Invariant I lean on everywhere:  d<x> has the EXACT same shape as <x>. If a
gradient's shape doesn't match, I forgot a sum (broadcast) or swapped a
transpose.

Run from the repo root with:
    python3 src/04-manual-backprop/main.py
"""

import torch
import torch.nn.functional as F

from common import display as ui
from common.data import load_words, build_vocab

SEED = 2147483647

# ----------------------------- Hyperparameters ------------------------------ #
BLOCK_SIZE = 3  # how many characters of context feed each prediction
N_EMBD = 20  # embedding dimensions per character
N_HIDDEN = 200  # neurons in the hidden layer
BATCH_SIZE = 32  # examples per minibatch (this is `n` in the backward formulas)
STEPS = 50000  # optimization steps (used when TRAIN is True)
LR = 0.07  # base lr
LR_FINE = LR / 10  # fine-tuning lr after the decay point

TRAIN = False  # gradient check runs by default; set True to train with the hand-derived grads.


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
    """Initialize the MLP + BatchNorm parameters.

    Note: the biases (b2, bngain, bnbias) are initialized to small *non-zero*
    random values on purpose. If they were exactly 0 / 1, a buggy backward could
    accidentally produce the right gradient and the cmp() check would pass for
    the wrong reason. Small random inits make the gradient check trustworthy.

    There is no b1: the hidden layer's bias is redundant with BatchNorm's bias.
    """
    C = torch.randn((vocab_size, N_EMBD), generator=g)
    W1 = (
        torch.randn((BLOCK_SIZE * N_EMBD, N_HIDDEN), generator=g)
        * (5 / 3)
        / ((BLOCK_SIZE * N_EMBD) ** 0.5)
    )
    W2 = torch.randn((N_HIDDEN, vocab_size), generator=g) * 0.1
    b2 = torch.randn(vocab_size, generator=g) * 0.1
    bngain = torch.randn((1, N_HIDDEN), generator=g) * 0.1 + 1.0
    bnbias = torch.randn((1, N_HIDDEN), generator=g) * 0.1

    params = [C, W1, W2, b2, bngain, bnbias]
    for p in params:
        p.requires_grad = True
    return params


def forward(params, Xb, Yb):
    """Forward pass, broken into atomic ops so every intermediate has a name.

    Returns (loss, cache) where cache holds every intermediate tensor the
    backward needs, both to backprop through and to compare against autograd.
    """
    C, W1, W2, b2, bngain, bnbias = params
    n = Xb.shape[0]

    # --- embedding lookup + concat ---
    emb = C[Xb]  # (n, block_size, n_embd)
    embcat = emb.view(n, -1)  # (n, block_size*n_embd)

    # --- hidden layer pre-activation (no bias: BatchNorm provides one) ---
    hprebn = embcat @ W1  # (n, n_hidden)

    # --- BatchNorm (kept as atomic steps; backward uses the compact formula) ---
    bnmeani = hprebn.mean(0, keepdim=True)  # (1, n_hidden)
    bndiff = hprebn - bnmeani  # (n, n_hidden)
    bnvar = bndiff.var(0, keepdim=True)  # (1, n_hidden)  unbiased (Bessel), torch default
    bnvar_inv = (bnvar + 1e-5) ** -0.5  # (1, n_hidden)
    bnraw = bndiff * bnvar_inv  # (n, n_hidden)
    hpreact = bngain * bnraw + bnbias  # (n, n_hidden)

    # --- non-linearity ---
    h = torch.tanh(hpreact)  # (n, n_hidden)

    # --- output layer ---
    logits = h @ W2 + b2  # (n, vocab_size)

    # --- loss ---
    loss = F.cross_entropy(logits, Yb)

    cache = dict(
        emb=emb,
        embcat=embcat,
        hprebn=hprebn,
        bnmeani=bnmeani,
        bndiff=bndiff,
        bnvar=bnvar,
        bnvar_inv=bnvar_inv,
        bnraw=bnraw,
        hpreact=hpreact,
        h=h,
        logits=logits,
    )
    return loss, cache


def manual_backward(params, Xb, Yb, c):
    """Hand-derived gradients, op by op.

    `c` is the forward cache (intermediate *values*, not gradients). Returns a
    dict {name: gradient} covering every tensor I check and every parameter I
    train with. Gradient rules I'm applying:
      - cross_entropy:  dlogits = (softmax(logits) - onehot(Y)) / n
      - linear h=x@W+b: dx = dh @ W.T ,  dW = x.T @ dh ,  db = dh.sum(0)
      - tanh:           dhpreact = (1 - h**2) * dh
      - matmul C=A@B:   dA = dC @ B.T ,  dB = A.T @ dC   (shapes pin the transposes)
      - view:           reshape the gradient back to emb's shape
      - indexing C[X]:  scatter-add (dC.index_add_) — a reused char accumulates
      - broadcast in forward  ==>  .sum over that axis in backward
      - compact BatchNorm backward:
            dhprebn = bngain * bnvar_inv / n * (
                n * dhpreact
                - dhpreact.sum(0)
                - n / (n - 1) * bnraw * (dhpreact * bnraw).sum(0)
            )
    """
    C, W1, W2, b2, bngain, bnbias = params
    n = Xb.shape[0]
    logits = c["logits"]
    h = c["h"]
    bnraw = c["bnraw"]
    bnvar_inv = c["bnvar_inv"]
    emb = c["emb"]
    embcat = c["embcat"]

    # 1. cross_entropy  ->  dlogits
    dlogits = None  # TODO

    # 2. output layer  logits = h @ W2 + b2  ->  dh, dW2, db2
    dh = None  # TODO
    dW2 = None  # TODO
    db2 = None  # TODO

    # 3. tanh  h = tanh(hpreact)  ->  dhpreact
    dhpreact = None  # TODO

    # 4. BatchNorm  hpreact = bngain * bnraw + bnbias  ->  dbngain, dbnbias, then dhprebn
    dbngain = None  # TODO
    dbnbias = None  # TODO
    dhprebn = None  # TODO  (compact formula above)

    # 5. hidden layer  hprebn = embcat @ W1  ->  dembcat, dW1
    dembcat = None  # TODO
    dW1 = None  # TODO

    # 6. view  embcat = emb.view(n, -1)  ->  demb
    demb = None  # TODO

    # 7. indexing  emb = C[Xb]  ->  dC  (scatter-add)
    dC = None  # TODO

    return {
        "logits": dlogits,
        "h": dh,
        "W2": dW2,
        "b2": db2,
        "hpreact": dhpreact,
        "bngain": dbngain,
        "bnbias": dbnbias,
        "hprebn": dhprebn,
        "embcat": dembcat,
        "W1": dW1,
        "emb": demb,
        "C": dC,
    }


def cmp(name, dt, t):
    """Compare a manual gradient `dt` to PyTorch's autograd `t.grad`.

    An unfilled gradient (still None) prints as TODO, so a run shows at a glance
    which gradients are done. The target is `approx: True` everywhere with a
    tiny maxdiff.
    """
    if dt is None:
        print(f"  {name:9s} | TODO")
        return
    exact = torch.all(dt == t.grad).item()
    approx = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    shape_ok = dt.shape == t.shape
    print(
        f"  {name:9s} | exact: {str(exact):5s} | approx: {str(approx):5s} "
        f"| shape ok: {str(shape_ok):5s} | maxdiff: {maxdiff:.3e}"
    )


def check_gradients(params, Xb, Yb):
    """Gradient check: one forward/backward pass, compare every manual grad to autograd."""
    C, W1, W2, b2, bngain, bnbias = params

    loss, c = forward(params, Xb, Yb)

    # Reference gradients from autograd. Keep grads on the intermediates too.
    for p in params:
        p.grad = None
    retained = ["logits", "h", "hpreact", "hprebn", "embcat", "emb"]
    for name in retained:
        c[name].retain_grad()
    loss.backward()

    grads = manual_backward(params, Xb, Yb, c)

    # The tensor each manual grad must match (intermediates -> cached, params -> leaf).
    targets = {
        "logits": c["logits"],
        "h": c["h"],
        "W2": W2,
        "b2": b2,
        "hpreact": c["hpreact"],
        "bngain": bngain,
        "bnbias": bnbias,
        "hprebn": c["hprebn"],
        "embcat": c["embcat"],
        "W1": W1,
        "emb": c["emb"],
        "C": C,
    }

    ui.kv("minibatch loss", f"{loss.item():.4f}  nats")
    print()
    for name, t in targets.items():
        cmp(name, grads[name], t)


def train(params, Xtr, Ytr, g):
    """Train using ONLY the hand-derived gradients — no loss.backward()."""
    decay_at = int(0.6 * STEPS)
    losses = []
    with ui.training_progress(STEPS) as step_done:
        for step in range(STEPS):
            ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,), generator=g)
            Xb, Yb = Xtr[ix], Ytr[ix]

            loss, c = forward(params, Xb, Yb)
            grads = manual_backward(params, Xb, Yb, c)

            lr = LR if step < decay_at else LR_FINE
            for p, name in zip(params, ["C", "W1", "W2", "b2", "bngain", "bnbias"]):
                p.data += -lr * grads[name]

            if step % 100 == 0 or step == STEPS - 1:
                losses.append(loss.item())
            step_done(loss=loss.item(), lr=lr)

    ui.kv("final loss", f"{losses[-1]:.4f}  nats  (last minibatch)")
    ui.loss_curve(losses, title="Training loss  (manual backprop)")


def main():
    g = torch.Generator().manual_seed(SEED)

    ui.banner(
        "MANUAL BACKPROPAGATION  (tensor-level)",
        f"MLP + BatchNorm, gradients by hand   (context = {BLOCK_SIZE} chars)",
    )

    _, itos, n_words, (Xtr, Ytr), (Xdev, Ydev), (Xte, Yte) = load_data(g)
    vocab_size = len(itos)

    ui.section("Dataset")
    ui.kv("words", f"{n_words:,}")
    ui.kv("vocabulary", f"{vocab_size} tokens  (a-z + '.')")
    ui.kv("train examples", f"{Xtr.shape[0]:,}")

    params = init_params(vocab_size, g)
    ui.section("Model")
    ui.kv("architecture", f"{BLOCK_SIZE}×{N_EMBD} → BN→tanh({N_HIDDEN}) → {vocab_size}")
    ui.kv("parameters", f"{sum(p.nelement() for p in params):,}")
    ui.kv("batch size (n)", BATCH_SIZE)

    # ----------------------------- Gradient check ------------------------------ #
    ui.section("Gradient Check  (manual backward vs autograd)")
    ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,), generator=g)
    check_gradients(params, Xtr[ix], Ytr[ix])

    # --------------------- Train with the manual gradients ---------------------- #
    if TRAIN:
        ui.section("Training  (manual gradients only)")
        train(params, Xtr, Ytr, g)


if __name__ == "__main__":
    main()
