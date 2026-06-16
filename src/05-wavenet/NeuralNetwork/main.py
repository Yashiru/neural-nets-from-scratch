import torch

class Sequential:
    def __init__(self, layers):
        self.layers = layers
        for p in self.parameters():
            p.requires_grad = True
    def __call__(self, x, y=None):
        self.target = y
        for layer in self.layers:
            x = layer(x)
        self.out = x
        if self.target is not None:
            self.loss = torch.nn.functional.cross_entropy(self.out, self.target)
        return self.loss, x
    def backward(self):
        self.zero_grad()
        if self.target is not None:
            self.loss.backward()
    def zero_grad(self):
        for layer in self.layers:
            for p in layer.parameters():
                if p.grad is not None:
                    p.grad.zero_()
    def optimize(self, lr):
        for layer in self.layers:
            for p in layer.parameters():
                p.data -= lr * p.grad
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    def train(self, mode=True):
        for layer in self.layers:
            layer.training = mode
    def eval(self):
        self.train(False)

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5   # init Kaiming
        self.bias = torch.zeros(fan_out) if bias else None
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out = self.out + self.bias
        return self.out
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    
class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return []
    
class BatchNorm1d:
    def __init__(self, n, momentum=0.1):
        self.momentum = momentum
        self.training = True
        # trained parameters
        self.gamma = torch.randn((1, n)) * 0.1 + 1.0
        self.beta = torch.randn((1, n)) * 0.1
        # running estimates (buffers, not trained) used at eval time
        self.running_mean = torch.zeros((1, n))
        self.running_var = torch.ones((1, n))
    def __call__(self, x):
        if self.training:
            dim = (0, 1) if x.ndim == 3 else 0
            mean = x.mean(dim, keepdim=True)
            var  = x.var(dim, keepdim=True)
        else:
            mean = self.running_mean
            var  = self.running_var
        self.out = (x - mean) / torch.sqrt(var + 1e-5) * self.gamma + self.beta
        if self.training:
            with torch.no_grad():  # update running stats without tracking grads
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
        return self.out
    def parameters(self):
        return [self.gamma, self.beta]
    
class embedding:
    def __init__(self, n, d):
        self.weight = torch.randn((n, d)) / d**0.5
    def __call__(self, x):
        self.out = self.weight[x]
        return self.out
    def parameters(self):
        return [self.weight]

class Flatten:
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out
    def parameters(self):
        return []
    
class FlattenConsecutive:
    def __init__(self, n):
        self.n = n                       # taille du groupe (2 ici)
    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)   # regrouper par n
        if x.shape[1] == 1:                       # plus qu'un groupe -> aplatir
            x = x.squeeze(1)
        self.out = x
        return x
    def parameters(self):
        return []