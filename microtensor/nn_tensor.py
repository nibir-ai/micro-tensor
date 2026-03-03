import numpy as np
from microtensor.tensor import Tensor
from microtensor.profiler import Profiler
from microtensor.ops_fast import fast_softmax, fast_gelu

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)
    def parameters(self): return []

# ==========================================
# TRANSFORMER ARCHITECTURE (Phase 3 & 4)
# ==========================================

class Linear(Module):
    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0 / in_features)
        self.w = Tensor(np.random.randn(in_features, out_features) * scale)
        self.b = Tensor(np.zeros((1, out_features)))
    
    @Profiler.profile
    def __call__(self, x):
        return x @ self.w + self.b
    
    def parameters(self): return [self.w, self.b]

class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = Tensor(np.ones((1, 1, dim)))
        self.beta = Tensor(np.zeros((1, 1, dim)))

    @Profiler.profile
    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean)**2).mean(axis=-1, keepdims=True)
        return self.gamma * ((x - mean) / (var + self.eps)**0.5) + self.beta
    
    def parameters(self): return [self.gamma, self.beta]

class MultiHeadAttention(Module):
    def __init__(self, embed_dim, num_heads):
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        # Fused projection: Q, K, and V in one matrix multiply
        self.qkv_proj = Linear(embed_dim, 3 * embed_dim) 
        self.out_proj = Linear(embed_dim, embed_dim)

    @Profiler.profile
    def __call__(self, x):
        N, L, D = x.data.shape
        H, d_k = self.num_heads, self.head_dim
        
        # 1. Fused Forward Projection
        qkv = self.qkv_proj(x).reshape((N, L, 3, H, d_k)).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 2. Scaled Dot-Product Attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(d_k))
        
        # 3. JIT Optimized Softmax
        orig_shape = scores.data.shape
        scores_flat = scores.data.reshape(-1, orig_shape[-1])
        probs_data = fast_softmax(scores_flat).reshape(orig_shape)
        probs = Tensor(probs_data, (scores,), 'fast_softmax')
        
        def _backward():
            # Softmax derivative approximation for the fused kernel
            scores.grad += (probs.data * (1.0 - probs.data)) * probs.grad 
        probs._backward = _backward

        # 4. Context Aggregation
        context = (probs @ v).transpose(0, 2, 1, 3).reshape((N, L, D))
        return self.out_proj(context)

    def parameters(self): return self.qkv_proj.parameters() + self.out_proj.parameters()

class TransformerBlock(Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln1, self.ln2 = LayerNorm(embed_dim), LayerNorm(embed_dim)
        self.w1, self.w2 = Linear(embed_dim, ff_dim), Linear(ff_dim, embed_dim)

    @Profiler.profile
    def __call__(self, x):
        # Attention + Residual
        x = self.ln1(x + self.attn(x))
        # FeedForward with fused fast_gelu
        ff_in = self.w1(x)
        ff_out = self.w2(Tensor(fast_gelu(ff_in.data), (ff_in,), 'fast_gelu'))
        # Final Norm + Residual
        return self.ln2(x + ff_out)

    def parameters(self):
        return self.attn.parameters() + self.ln1.parameters() + \
               self.ln2.parameters() + self.w1.parameters() + self.w2.parameters()

class LanguageModel(Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_len):
        self.token_emb = Embedding(vocab_size, embed_dim)
        self.pos_emb = PositionalEncoding(max_seq_len, embed_dim)
        self.blocks = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        self.ln_f = LayerNorm(embed_dim)
        self.head = Linear(embed_dim, vocab_size)

    @Profiler.profile
    def __call__(self, idx):
        x = self.pos_emb(self.token_emb(idx))
        for block in self.blocks: x = block(x)
        return self.head(self.ln_f(x))

    def parameters(self):
        p = self.token_emb.parameters() + self.ln_f.parameters() + self.head.parameters()
        for b in self.blocks: p += b.parameters()
        return p

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
        pos = np.arange(max_seq_len).reshape(-1, 1)
        div = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        pe[:, 0::2], pe[:, 1::2] = np.sin(pos * div), np.cos(pos * div)
        self.pe = Tensor(pe)
    
    def __call__(self, x): 
        return x + self.pe.data[:x.data.shape[1], :]

def cross_entropy(logits, target_indices):
    """Numerically stable Cross-Entropy for Sequence tasks."""
    N, L, V = logits.data.shape
    logits_flat = logits.reshape((N * L, V))
    targets_flat = target_indices.flatten()
    max_val = Tensor(np.max(logits_flat.data, axis=1, keepdims=True))
    
    log_sum_exp = (logits_flat - max_val).exp().sum(axis=1, keepdims=True).log()
    log_probs = (logits_flat - max_val) - log_sum_exp
    
    target_one_hot = np.zeros_like(logits_flat.data)
    target_one_hot[np.arange(N*L), targets_flat] = 1.0
    return (Tensor(target_one_hot) * log_probs).sum() * (-1.0 / (N * L))


# ==========================================
# LEGACY ARCHITECTURE (Phase 1 & 2 Tests)
# ==========================================

class MLP(Module):
    def __init__(self, nin, nouts):
        self.layers = [Linear(nin, nouts[0])] + [Linear(nouts[i], nouts[i+1]) for i in range(len(nouts)-1)]
    
    @Profiler.profile
    def __call__(self, x):
        for layer in self.layers: x = layer(x).tanh()
        return x
        
    def parameters(self): return [p for l in self.layers for p in l.parameters()]

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        scale = np.sqrt(2.0 / (in_channels * kernel_size**2))
        self.weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale)
        self.bias = Tensor(np.zeros((1, out_channels, 1, 1)))

    @Profiler.profile
    def __call__(self, x):
        N, C, H, W = x.data.shape
        K, S, P = self.kernel_size, self.stride, self.padding
        x_pad = np.pad(x.data, ((0,0), (0,0), (P,P), (P,P)), mode='constant') if P > 0 else x.data
        out_h, out_w = (H + 2*P - K) // S + 1, (W + 2*P - K) // S + 1
        shape = (N, C, out_h, out_w, K, K)
        strides = (x_pad.strides[0], x_pad.strides[1], S*x_pad.strides[2], S*x_pad.strides[3], x_pad.strides[2], x_pad.strides[3])
        patches = np.lib.stride_tricks.as_strided(x_pad, shape=shape, strides=strides)
        input_matrix = Tensor(patches.transpose(0, 2, 3, 1, 4, 5).reshape(N, out_h*out_w, -1))
        kernels = self.weight.reshape((self.out_channels, -1))
        res = input_matrix @ kernels.transpose(1, 0)
        return res.transpose(0, 2, 1).reshape((N, self.out_channels, out_h, out_w)) + self.bias
        
    def parameters(self): return [self.weight, self.bias]

class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        
    @Profiler.profile
    def __call__(self, x):
        N, C, H, W = x.data.shape
        K, S = self.kernel_size, self.stride
        out_h, out_w = H // S, W // S
        shape = (N, C, out_h, out_w, K, K)
        strides = (x.data.strides[0], x.data.strides[1], S*x.data.strides[2], S*x.data.strides[3], x.data.strides[2], x.data.strides[3])
        patches = np.lib.stride_tricks.as_strided(x.data, shape=shape, strides=strides)
        res_data = np.max(patches, axis=(4, 5))
        out = Tensor(res_data, (x,), 'maxpool')
        def _backward():
            mask = (patches == res_data[:, :, :, :, None, None])
            grad_expanded = out.grad[:, :, :, :, None, None] * mask
            for i in range(out_h):
                for j in range(out_w):
                    x.grad[:, :, i*S:i*S+K, j*S:j*S+K] += grad_expanded[:, :, i, j, :, :]
        out._backward = _backward
        return out
        
    def parameters(self): return []

class Flatten(Module):
    def __call__(self, x): return x.reshape((x.data.shape[0], -1))
