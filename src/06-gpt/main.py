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
from common.data import load_words, build_vocab, load_text, build_char_vocab
from common.NeuralNetwork.GPT import GPT

# ----------------------------- Hyperparameters ------------------------------ #
BLOCK_SIZE = 32  # how many characters of context feed each prediction
N_EMBD = 256  # embedding dimensions per character
N_HIDDEN = 512  # neurons in the hidden layer
BATCH_SIZE = 256  # examples per minibatch (this is `n` in the backward formulas)
STEPS = 40_000  # optimization steps (used when TRAIN is True)
LR = .1
LR_FINE = LR / 10  # fine-tuning lr after the decay point
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # train on the GPU when available
DATASET = "shakespeare"  # which corpus to train on: "names" | "shakespeare"
USE_AMP = False  # mixed precision. Measured: it only pays off on big models
# (n_embd >= ~384, long context, large batch); on small models the autocast cast
# overhead makes each step SLOWER. Flip to True once the model is large enough.


def main():
    _, itos, n_items, (Xtr, Ytr), (Xdev, Ydev), (Xte, Yte) = load_data(DATASET)
    # X holds (N, BLOCK_SIZE) context windows; Y holds the same windows shifted by
    # one position, so every position carries its next-token target (N, BLOCK_SIZE).
    vocab_size = len(itos)

    model = GPT(vocab_size, N_EMBD, n_head=4, n_layer=4)
    model.to(DEVICE)
    amp_dtype = pick_amp_dtype() if USE_AMP else None

    ui.banner(
        "DECODER-ONLY TRANSFORMER  (GPT from scratch)",
        f"masked self-attention, autograd + manual optimizer   (context = {BLOCK_SIZE} chars)",
    )

    units = "characters" if DATASET == "shakespeare" else "words"
    ui.section("Dataset")
    ui.kv("corpus", DATASET)
    ui.kv("size", f"{n_items:,} {units}")
    ui.kv("vocabulary", f"{vocab_size} tokens")
    ui.kv("train examples", f"{Xtr.shape[0]:,}")

    ui.section("Model")
    ui.kv(
        "architecture",
        f"GPT(vocab_size={vocab_size}, n_embd={N_EMBD}, n_head=4, n_layer=4)",
    )
    ui.kv("parameters", f"{sum(p.nelement() for p in model.parameters()):,}")
    ui.kv("batch size (n)", BATCH_SIZE)
    ui.kv("device", DEVICE)
    ui.kv("precision", str(amp_dtype).rsplit(".", 1)[-1] if amp_dtype else "float32")

    # ------------------------------- Train the model ---------------------------- #
    ui.section("Training")
    train(model, Xtr, Ytr, amp_dtype)

    # --------------------------- Held-out loss estimates ------------------------ #
    ui.section("Evaluation")
    ui.kv("dev loss", f"{evaluate_loss(model, Xdev, Ydev):.4f}  nats")
    ui.kv("test loss", f"{evaluate_loss(model, Xte, Yte):.4f}  nats")

    # ------------------------------- Sample the model --------------------------- #
    if DATASET == "names":
        samples = [
            generate(model, itos, BLOCK_SIZE, max_new_tokens=40, stop_token=0)
            for _ in range(20)
        ]
        first = next((s for s in samples if s[0]), samples[0])
        ui.section("Sampling Trace  (first generated name)")
        ui.trace(first[1])
        ui.section(f"Generated Names  ({len(samples)} samples)")
        ui.name_list([s for s in samples if s[0]])
    else:
        text, _ = generate(model, itos, BLOCK_SIZE, max_new_tokens=500)
        ui.section("Generated Text  (500 characters)")
        print(text)


def pick_amp_dtype():
    """Pick the fastest safe autocast dtype for the current GPU (None -> full FP32).

    Ampere+ (sm_80+) has bf16 tensor cores and needs no loss scaling; Volta/Turing
    (sm_70/75) has fp16 tensor cores (scaling required); older GPUs / CPU stay FP32.
    NB: torch.cuda.is_bf16_supported() reports True on Turing via (unaccelerated)
    emulation, so we gate on the compute capability instead.
    """
    if not torch.cuda.is_available():
        return None
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        return torch.bfloat16
    if major >= 7:
        return torch.float16
    return None


def train(model, Xtr, Ytr, amp_dtype=None):
    """Train with autograd + the hand-rolled optimizer, optionally in mixed precision."""
    decay_at = int(0.6 * STEPS)
    losses = []
    Xtr, Ytr = Xtr.to(DEVICE), Ytr.to(DEVICE)  # move the whole dataset to the GPU once
    # fp16 backward underflows tiny gradients to zero; a fixed loss scale lifts them
    # above the fp16 floor, then we divide it back out. (bf16 and fp32 need no scaling.)
    loss_scale = 1024.0 if amp_dtype == torch.float16 else 1.0
    with ui.training_progress(STEPS) as step_done:
        for step in range(STEPS):
            ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,), device=DEVICE)
            Xb, Yb = Xtr[ix], Ytr[ix]
            lr = LR_FINE if step >= decay_at else LR

            if amp_dtype is not None:
                with torch.autocast(device_type=DEVICE, dtype=amp_dtype):
                    logits = model(Xb)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Yb.view(-1))
            else:
                logits = model(Xb)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Yb.view(-1))

            model.zero_grad()
            (loss * loss_scale).backward()
            if loss_scale != 1.0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(1.0 / loss_scale)
            model.optimize(lr=lr)

            if step % 100 == 0 or step == STEPS - 1:
                losses.append(loss.item())
                step_done(loss=loss.item(), lr=lr)
            else:
                step_done(loss=losses[-1], lr=lr, force=True)

    ui.kv("final loss", f"{losses[-1]:.4f}  nats  (last minibatch)")
    ui.kv("initial loss", f"{losses[0]:.4f}  nats")
    ui.loss_curve(losses, title="Training loss")


@torch.no_grad()
def evaluate_loss(model, X, Y, batch_size=512):
    """Average cross-entropy over a whole split, computed in batches on the device."""
    X, Y = X.to(DEVICE), Y.to(DEVICE)
    total, count = 0.0, 0
    for i in range(0, X.shape[0], batch_size):
        xb, yb = X[i : i + batch_size], Y[i : i + batch_size]
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        total += loss.item() * xb.shape[0]
        count += xb.shape[0]
    return total / count


@torch.no_grad()
def generate(model, itos, block_size, max_new_tokens=40, stop_token=None):
    context = [0] * block_size
    out, steps = [], []
    for _ in range(max_new_tokens):
        logits = model(torch.tensor([context], device=DEVICE))  # (1, T, vocab_size)
        logits = logits[:, -1, :]  # keep only the last step's prediction -> (1, vocab_size)

        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1).item()
        ctx = "".join(itos[i] for i in context)
        steps.append((ctx, itos[ix], float(probs[0, ix])))
        context = context[1:] + [ix]
        if ix == stop_token:  # reached the end-of-sequence token (names use '.')
            break
        out.append(itos[ix])
    return "".join(out), steps


def load_data(dataset):
    """Dispatch to the loader for the selected corpus."""
    if dataset == "names":
        return load_names_data()
    if dataset == "shakespeare":
        return load_text_data()
    raise ValueError(f"unknown dataset: {dataset!r} (expected 'names' or 'shakespeare')")


def load_names_data():
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


def load_text_data(filename="tinyshakespeare.txt"):
    """Load a text corpus, encode it to one stream, and split it 80/10/10."""
    text = load_text(filename)
    stoi, itos = build_char_vocab(text)
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n1, n2 = int(0.8 * len(data)), int(0.9 * len(data))
    tr = build_text_dataset(data[:n1], BLOCK_SIZE)
    dev = build_text_dataset(data[n1:n2], BLOCK_SIZE)
    te = build_text_dataset(data[n2:], BLOCK_SIZE)
    return stoi, itos, len(text), tr, dev, te


def build_dataset(words, stoi, block_size):
    """Turn words into (window, next-token-per-position) tensors, both (N, block_size)."""
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            context = context[1:] + [ix]
            Y.append(context)
    return torch.tensor(X), torch.tensor(Y)


def build_text_dataset(stream, block_size):
    """Slice a continuous token stream into (window, next-token-per-position) tensors.

    Y is the window shifted one position right, so each position predicts the
    following token. Both X and Y have shape (len(stream) - block_size, block_size).
    """
    X = stream[:-1].unfold(0, block_size, 1).contiguous()
    Y = stream[1:].unfold(0, block_size, 1).contiguous()
    return X, Y


if __name__ == "__main__":
    main()
