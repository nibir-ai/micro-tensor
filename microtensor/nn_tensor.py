import numpy as np
from microtensor.tensor import Tensor

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)
    def parameters(self): return []

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.w = Tensor(np.random.randn(in_features, out_features) * scale)
        self.b = Tensor(np.zeros((1, out_features))) if bias else None
    def __call__(self, x):
        out = x @ self.w
        if self.b: out = out + self.b
        return out
    def parameters(self): return [self.w] + ([self.b] if self.b else [])

class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = Tensor(np.ones((1, 1, dim)))
        self.beta = Tensor(np.zeros((1, 1, dim)))
    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean)**2).mean(axis=-1, keepdims=True)
        x_hat = (x - mean) / (var + self.eps)**0.5
        return self.gamma * x_hat + self.beta
    def parameters(self): return [self.gamma, self.beta]

class Embedding(Module):
    def __init__(self, vocab_size, embed_dim):
        self.weight = Tensor(np.random.randn(vocab_size, embed_dim) * 0.1)
    def __call__(self, indices):
        idx = np.array(indices)
        out = Tensor(self.weight.data[idx], (self.weight,), 'embedding')
        def _backward(): np.add.at(self.weight.grad, idx, out.grad)
        out._backward = _backward
        return out
    def parameters(self): return [self.weight]

class PositionalEncoding(Module):
    def __init__(self, max_seq_len, embed_dim):
        pe = np.zeros((max_seq_len, embed_dim))
        position = np.arange(max_seq_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        pe[:, 0::2], pe[:, 1::2] = np.sin(position * div_term), np.cos(position * div_term)
        self.pe = Tensor(pe)
    def __call__(self, x): return x + self.pe.data[:x.data.shape[1], :]

class MultiHeadAttention(Module):
    def __init__(self, embed_dim, num_heads):
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)
    def __call__(self, x):
        N, L, D = x.data.shape
        H, d_k = self.num_heads, self.head_dim
        q = self.q_proj(x).reshape((N, L, H, d_k)).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape((N, L, H, d_k)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape((N, L, H, d_k)).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(d_k))
        exp_scores = (scores - Tensor(np.max(scores.data, axis=-1, keepdims=True))).exp()
        probs = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
        context = (probs @ v).transpose(0, 2, 1, 3).reshape((N, L, D))
        return self.out_proj(context)
    def parameters(self):
        return self.q_proj.parameters() + self.k_proj.parameters() + self.v_proj.parameters() + self.out_proj.parameters()

class TransformerBlock(Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)
        self.ff = MLP(embed_dim, [ff_dim, embed_dim])
    def __call__(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ff(x))
        return x
    def parameters(self):
        return self.attn.parameters() + self.ln1.parameters() + self.ln2.parameters() + self.ff.parameters()

class LanguageModel(Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_len):
        self.token_emb = Embedding(vocab_size, embed_dim)
        self.pos_emb = PositionalEncoding(max_seq_len, embed_dim)
        self.blocks = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        self.ln_f = LayerNorm(embed_dim)
        self.head = Linear(embed_dim, vocab_size)
        self.max_seq_len = max_seq_len

    def __call__(self, idx):
        x = self.token_emb(idx)
        x = self.pos_emb(x)
        for block in self.blocks: x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    def generate(self, idx, max_new_tokens):
        # idx is (1, T) array of indices
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            idx_cond = idx[:, -self.max_seq_len:]
            # Forward to get logits
            logits = self(idx_cond)
            # Focus only on the last time step
            logits = logits.data[:, -1, :] # (1, Vocab)
            # Sample (Greedy for now)
            next_idx = np.argmax(logits, axis=-1, keepdims=True)
            # Append to sequence
            idx = np.concatenate([idx, next_idx], axis=1)
        return idx

    def parameters(self):
        params = self.token_emb.parameters() + self.head.parameters() + self.ln_f.parameters()
        for block in self.blocks: params += block.parameters()
        return params

class MLP(Module):
    def __init__(self, nin, nouts):
        self.layers = [Linear(nin, nouts[0])] + [Linear(nouts[i], nouts[i+1]) for i in range(len(nouts)-1)]
    def __call__(self, x):
        for layer in self.layers: x = layer(x).tanh()
        return x
    def parameters(self): return [p for l in self.layers for p in l.parameters()]

class Flatten(Module):
    def __call__(self, x): return x.reshape((x.data.shape[0], -1))

def cross_entropy(logits, target_indices):
    N, L, V = logits.data.shape
    logits_flat = logits.reshape((N * L, V))
    targets_flat = target_indices.flatten()
    max_val = Tensor(np.max(logits_flat.data, axis=1, keepdims=True))
    logits_stable = logits_flat - max_val
    log_probs = logits_stable - logits_stable.exp().sum(axis=1, keepdims=True).log()
    target_one_hot = np.zeros_like(logits_flat.data)
    target_one_hot[np.arange(N*L), targets_flat] = 1.0
    return (Tensor(target_one_hot) * log_probs).sum() * (-1.0 / (N * L))
