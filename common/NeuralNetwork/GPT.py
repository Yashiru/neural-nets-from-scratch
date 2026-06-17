import torch
import torch.nn.functional as F
from common.NeuralNetwork.basics import LayerNorm, Sequential, Linear, ReLu


class Head:
    def __init__(self, head_size, n_embd, context_size=1024):
        self.key = torch.nn.Linear(n_embd, head_size, bias=False)
        self.query = torch.nn.Linear(n_embd, head_size, bias=False)
        self.value = torch.nn.Linear(n_embd, head_size, bias=False)
        self.tril = torch.tril(torch.ones(context_size, context_size))

    def __call__(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        # check shape
        assert k.shape == (B, T, self.key.out_features), (
            f"key shape mismatch: {k.shape} != {(B, T, self.key.out_features)}"
        )
        q = self.query(x)  # (B, T, head_size)
        # check shape
        assert q.shape == (B, T, self.query.out_features), (
            f"query shape mismatch: {q.shape} != {(B, T, self.query.out_features)}"
        )
        wei = (
            q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        )  # (B, T, T), on scale dot-product attention
        # check shape
        assert wei.shape == (B, T, T), (
            f"attention weights shape mismatch: {wei.shape} != {(B, T, T)}"
        )
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # casual masking
        wei = F.softmax(wei, dim=-1)
        # check shape
        assert wei.shape == (B, T, T), (
            f"softmax attention weights shape mismatch: {wei.shape} != {(B, T, T)}"
        )
        v = self.value(x)  # (B, T, head_size)
        # check shape
        assert v.shape == (B, T, self.value.out_features), (
            f"value shape mismatch: {v.shape} != {(B, T, self.value.out_features)}"
        )
        out = wei @ v  # (B, T, head_size)
        # check shape
        assert out.shape == (B, T, self.value.out_features), (
            f"output shape mismatch: {out.shape} != {(B, T, self.value.out_features)}"
        )
        return out

    def parameters(self):
        return (
            list(self.key.parameters())
            + list(self.query.parameters())
            + list(self.value.parameters())
        )


class MultiHeadAttention:
    def __init__(self, num_heads, head_size, n_embd, context_size=1024):
        self.heads = [Head(head_size, n_embd, context_size) for _ in range(num_heads)]
        self.proj = torch.nn.Linear(num_heads * head_size, n_embd)

    def __call__(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

    def parameters(self):
        return [param for head in self.heads for param in head.parameters()] + list(
            self.proj.parameters()
        )


class Block:
    def __init__(self, n_embd, n_head, context_size=1024):
        self.sa = MultiHeadAttention(n_head, n_embd // n_head, n_embd, context_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

    def __call__(self, x):
        x = x + self.sa(self.ln1(x))  # communication (pre-norm + résiduel)
        x = x + self.ffwd(self.ln2(x))  # computation  (pre-norm + résiduel)
        return x

    def parameters(self):
        return (
            list(self.sa.parameters())
            + list(self.ffwd.parameters())
            + list(self.ln1.parameters())
            + list(self.ln2.parameters())
        )


class FeedForward:
    def __init__(self, n_embd):
        self.net = Sequential(
            [
                Linear(n_embd, 4 * n_embd),
                ReLu(),
                Linear(4 * n_embd, n_embd),
            ]
        )

    def __call__(self, x):
        loss, x = self.net(x)
        return x

    def parameters(self):
        return self.net.parameters()


class Transformer:
    def __init__(self, n_embd, n_head, n_layer, context_size=1024):
        self.blocks = [Block(n_embd, n_head, context_size) for _ in range(n_layer)]

    def __call__(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def parameters(self):
        return [param for block in self.blocks for param in block.parameters()]


class GPT:
    def __init__(self, vocab_size, n_embd, n_head, n_layer, context_size=1024):
        self.token_embedding_table = torch.nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = torch.nn.Embedding(context_size, n_embd)
        self.blocks = Transformer(n_embd, n_head, n_layer, context_size)
        self.ln_f = LayerNorm(n_embd)
        self.lm_head = torch.nn.Linear(n_embd, vocab_size)

    def __call__(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        return logits

    def parameters(self):
        return (
            list(self.token_embedding_table.parameters())
            + list(self.position_embedding_table.parameters())
            + list(self.blocks.parameters())
            + list(self.ln_f.parameters())
            + list(self.lm_head.parameters())
        )

    def zero_grad(self):
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def optimize(self, lr):
        for param in self.parameters():
            if param.grad is not None:
                param.data -= lr * param.grad
