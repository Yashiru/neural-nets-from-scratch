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
        # assert k.shape == (B, T, self.key.out_features), (
        #     f"key shape mismatch: {k.shape} != {(B, T, self.key.out_features)}"
        # )
        q = self.query(x)  # (B, T, head_size)
        # check shape
        # assert q.shape == (B, T, self.query.out_features), (
        #     f"query shape mismatch: {q.shape} != {(B, T, self.query.out_features)}"
        # )
        wei = (
            q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        )  # (B, T, T), scaled dot-product attention
        # check shape
        # assert wei.shape == (B, T, T), (
        #     f"attention weights shape mismatch: {wei.shape} != {(B, T, T)}"
        # )
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # causal masking
        wei = F.softmax(wei, dim=-1)
        # check shape
        # assert wei.shape == (B, T, T), (
        #     f"softmax attention weights shape mismatch: {wei.shape} != {(B, T, T)}"
        # )
        v = self.value(x)  # (B, T, head_size)
        # check shape
        # assert v.shape == (B, T, self.value.out_features), (
        #     f"value shape mismatch: {v.shape} != {(B, T, self.value.out_features)}"
        # )
        out = wei @ v  # (B, T, head_size)
        # check shape
        # assert out.shape == (B, T, self.value.out_features), (
        #     f"output shape mismatch: {out.shape} != {(B, T, self.value.out_features)}"
        # )
        return out

    def parameters(self):
        return (
            list(self.key.parameters())
            + list(self.query.parameters())
            + list(self.value.parameters())
        )

    def to(self, device):
        self.key.to(device)
        self.query.to(device)
        self.value.to(device)
        self.tril = self.tril.to(device)  # buffer (not a parameter) -> must be moved manually
        return self

class OptimizedMultiHeadAttention:
    def __init__(self, num_heads, head_size, n_embd, context_size=1024):
        self.num_heads = num_heads
        self.head_size = head_size
        # one projection per q/k/v, covering ALL heads at once
        self.key   = torch.nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.query = torch.nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.value = torch.nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.proj  = torch.nn.Linear(num_heads * head_size, n_embd)
        self.tril  = torch.tril(torch.ones(context_size, context_size))

    def __call__(self, x):
        B, T, C = x.shape
        nh, hs = self.num_heads, self.head_size
        # (B,T,C) -> (B,T,nh,hs) -> (B,nh,T,hs): move the heads into the batch dim
        k = self.key(x).view(B, T, nh, hs).transpose(1, 2)
        q = self.query(x).view(B, T, nh, hs).transpose(1, 2)
        v = self.value(x).view(B, T, nh, hs).transpose(1, 2)

        wei = q @ k.transpose(-2, -1) * (hs ** -0.5)          # (B,nh,T,T), a single batched matmul
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v                                          # (B,nh,T,hs)
        out = out.transpose(1, 2).contiguous().view(B, T, nh * hs)  # recombine the heads
        return self.proj(out)

    def parameters(self):
        return (list(self.key.parameters()) + list(self.query.parameters())
                + list(self.value.parameters()) + list(self.proj.parameters()))

    def to(self, device):
        self.key.to(device); self.query.to(device)
        self.value.to(device); self.proj.to(device)
        self.tril = self.tril.to(device)
        return self

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

    def to(self, device):
        for h in self.heads:
            h.to(device)
        self.proj.to(device)
        return self


class Block:
    def __init__(self, n_embd, n_head, context_size=1024):
        self.sa = OptimizedMultiHeadAttention(n_head, n_embd // n_head, n_embd, context_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

    def __call__(self, x):
        x = x + self.sa(self.ln1(x))  # communication (pre-norm + residual)
        x = x + self.ffwd(self.ln2(x))  # computation  (pre-norm + residual)
        return x

    def parameters(self):
        return (
            list(self.sa.parameters())
            + list(self.ffwd.parameters())
            + list(self.ln1.parameters())
            + list(self.ln2.parameters())
        )

    def to(self, device):
        self.sa.to(device)
        self.ffwd.to(device)
        self.ln1.to(device)
        self.ln2.to(device)
        return self


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

    def to(self, device):
        self.net.to(device)
        return self


class Transformer:
    def __init__(self, n_embd, n_head, n_layer, context_size=1024):
        self.blocks = [Block(n_embd, n_head, context_size) for _ in range(n_layer)]

    def __call__(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def parameters(self):
        return [param for block in self.blocks for param in block.parameters()]

    def to(self, device):
        for block in self.blocks:
            block.to(device)
        return self


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

    def to(self, device):
        self.token_embedding_table.to(device)
        self.position_embedding_table.to(device)
        self.blocks.to(device)
        self.ln_f.to(device)
        self.lm_head.to(device)
        return self

    def zero_grad(self):
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()
    
    def optimized_zero_grad(self):
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        if grads:
            torch._foreach_zero_(grads)            # zero all grads at once, fused

    def optimize(self, lr):
        for param in self.parameters():
            if param.grad is not None:
                param.data -= lr * param.grad
    
    def optimized_optimize(self, lr):
        datas, grads = [], []
        for p in self.parameters():
            if p.grad is not None:
                datas.append(p.data)               # .data: bypass autograd, like before
                grads.append(p.grad)
        torch._foreach_add_(datas, grads, alpha=-lr)   # data[i] += -lr * grad[i], fused
