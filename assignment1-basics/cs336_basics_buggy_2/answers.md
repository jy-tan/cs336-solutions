# ML Debugging Practice - Answer Key

This document contains all 8 bugs introduced in the `cs336_basics_buggy_2` codebase, along with explanations of why they're wrong and how to fix them.

**Difficulty Rating**: ⭐ Easy | ⭐⭐ Medium | ⭐⭐⭐ Hard

---

## Bug 1: Softmax Dimension Hardcoded ⭐⭐

**File**: `transformer/core.py`  
**Function**: `softmax()`  
**Line**: ~270

### The Bug
```python
# BUGGY:
return exp_x / exp_x.sum(-1, keepdim=True)
```

### Why It's Wrong
The `softmax` function takes a `dimension` parameter that specifies which axis to normalize over. However, the denominator sum is hardcoded to dimension `-1` instead of using the `dimension` parameter. This means softmax will always normalize over the last dimension, regardless of what the caller requests.

### The Fix
```python
# CORRECT:
return exp_x / exp_x.sum(dimension, keepdim=True)
```

### Symptom
This bug may not cause immediate crashes but will produce incorrect attention scores if softmax is ever called with a dimension other than -1. The numerical results will be subtly wrong, leading to poor model performance.

---

## Bug 2: Attention Mask Logic Inverted ⭐⭐⭐

**File**: `transformer/attention.py`  
**Function**: `scaled_dot_product_attention()`  
**Line**: ~48

### The Bug
```python
# BUGGY:
scaled_qk_dot_product = scaled_qk_dot_product.masked_fill(mask, -float("inf"))
```

### Why It's Wrong
The mask uses the convention where `True` means "keep this position" and `False` means "mask it out". The `masked_fill` function sets values to `-inf` where the mask is `True`. By passing `mask` directly (instead of `~mask`), we're masking out the positions we want to KEEP and keeping the positions we want to MASK.

For causal attention, this means the model can see future tokens but NOT past tokens - the exact opposite of what we want!

### The Fix
```python
# CORRECT:
scaled_qk_dot_product = scaled_qk_dot_product.masked_fill(~mask, -float("inf"))
```

### Symptom
The model will have "anti-causal" attention - it can only attend to future positions, not past ones. This completely breaks autoregressive generation and will cause the model to produce nonsensical outputs.

---

## Bug 3: Attention Scaling Inverted ⭐⭐

**File**: `transformer/attention.py`  
**Function**: `scaled_dot_product_attention()`  
**Line**: ~44

### The Bug
```python
# BUGGY:
scaled_qk_dot_product = qk_dot_product * (d_k**0.5)
```

### Why It's Wrong
Scaled dot-product attention divides by √d_k to prevent the dot products from growing too large in magnitude. When d_k is large, the dot products can become very large, pushing softmax into regions where it has extremely small gradients (saturated softmax).

The bug multiplies by √d_k instead of dividing, making the problem exponentially worse!

### The Fix
```python
# CORRECT:
scaled_qk_dot_product = qk_dot_product / (d_k**0.5)
```

### Symptom
- Extremely peaked softmax distributions (one attention weight ≈ 1, others ≈ 0)
- Vanishing gradients during training
- Training instability or NaN losses
- The model may appear to "work" but will learn very slowly or not at all

---

## Bug 4: RMSNorm Epsilon Placement ⭐⭐⭐

**File**: `transformer/core.py`  
**Function**: `RMSNorm.forward()`  
**Line**: ~109

### The Bug
```python
# BUGGY:
rms_inv = torch.rsqrt(reduce(x**2, "... d_model -> ... 1", "mean")) + self.eps
```

### Why It's Wrong
The epsilon (small constant for numerical stability) should be added INSIDE the `rsqrt()` operation, not outside. The purpose of epsilon is to prevent division by zero when the RMS is very small. Adding it after `rsqrt()` doesn't prevent the division-by-zero - it just adds a small constant to the result.

Furthermore, `rsqrt(x) + eps ≠ rsqrt(x + eps)`. The math is completely different:
- `rsqrt(0) + eps = inf + eps = inf` (still broken!)
- `rsqrt(0 + eps) = rsqrt(eps) = finite` (works correctly)

### The Fix
```python
# CORRECT:
rms_inv = torch.rsqrt(reduce(x**2, "... d_model -> ... 1", "mean") + self.eps)
```

### Symptom
- NaN or Inf values when input has very small magnitude
- Training may crash randomly depending on input distribution
- Can be intermittent and hard to reproduce

---

## Bug 5: RoPE Rotation Formula Error ⭐⭐⭐

**File**: `transformer/core.py`  
**Function**: `RotaryPositionalEmbedding.forward()`  
**Line**: ~232-233

### The Bug
```python
# BUGGY:
row1 = x1 * cos - x2 * sin
row2 = x2 * sin + x1 * cos  # Wrong! Should be x1 * sin + x2 * cos
```

### Why It's Wrong
Rotary Position Embedding applies a 2D rotation matrix to pairs of dimensions:
```
[cos θ  -sin θ] [x1]   [x1 cos θ - x2 sin θ]
[sin θ   cos θ] [x2] = [x1 sin θ + x2 cos θ]
```

The bug computes `row2 = x2 * sin + x1 * cos` instead of `x1 * sin + x2 * cos`. This isn't a valid rotation - the terms are in the wrong order.

### The Fix
```python
# CORRECT:
row1 = x1 * cos - x2 * sin
row2 = x1 * sin + x2 * cos
```

### Symptom
- Positional information will be corrupted
- The model won't properly distinguish between positions
- Attention patterns will be wrong, especially for longer sequences
- Model may still train but will have degraded performance on tasks requiring positional understanding

---

## Bug 6: Missing Residual Connection ⭐

**File**: `transformer/transformer.py`  
**Function**: `TransformerBlock.forward()`  
**Line**: ~54-58

### The Bug
```python
# BUGGY:
# Multi-head attention sublayer
x = self.ln1(x)
x = self.attn(x, token_positions=token_positions)  # Missing + x_orig!

# Feed-forward sublayer
x_orig = x.clone()
x = self.ln2(x)
x = self.ffn(x) + x_orig
```

### Why It's Wrong
The Transformer architecture uses residual (skip) connections around both the attention and FFN sublayers. The pattern is:
```
x = sublayer(norm(x)) + x
```

The bug removes the residual connection from the attention sublayer. Without residual connections:
1. Gradients can't flow directly through the network (vanishing gradients)
2. The network loses the ability to easily learn identity mappings
3. Deep networks become nearly impossible to train

### The Fix
```python
# CORRECT:
x_orig = x.clone()

# Multi-head attention sublayer
x = self.ln1(x)
x = self.attn(x, token_positions=token_positions) + x_orig

# Feed-forward sublayer
x_orig = x.clone()
x = self.ln2(x)
x = self.ffn(x) + x_orig
```

### Symptom
- Very poor gradient flow, especially in deep networks
- Training loss may not decrease
- Model outputs may collapse or become uniform

---

## Bug 7: Cross-Entropy Loss Sign Error ⭐⭐

**File**: `training/loss.py`  
**Function**: `cross_entropy_loss()`  
**Line**: ~34

### The Bug
```python
# BUGGY:
loss = target_log_probs.mean()
```

### Why It's Wrong
Cross-entropy loss is defined as the NEGATIVE log probability of the correct class. Log probabilities are always ≤ 0 (since probabilities are ≤ 1). 

Without the negative sign:
- The loss will be negative (or zero)
- Gradient descent will try to MINIMIZE this, meaning it will try to make the loss MORE negative
- This means maximizing log probability... wait, that's actually correct?

Actually, let me reconsider. The gradient descent update is: `θ = θ - lr * ∇L`

With correct loss `L = -log(p)`:
- We minimize L, which means maximizing log(p), which means maximizing p ✓

With buggy loss `L = +log(p)`:
- We minimize L, which means minimizing log(p), which means minimizing p ✗
- The model will learn to assign LOWER probability to the correct tokens!

### The Fix
```python
# CORRECT:
loss = -target_log_probs.mean()
```

### Symptom
- Loss will start negative and become more negative
- Model will actively learn to predict the WRONG tokens
- Perplexity will increase instead of decrease

---

## Bug 8: AdamW Weight Decay Order ⭐⭐⭐

**File**: `training/optimizer.py`  
**Function**: `AdamW.step()`  
**Line**: ~82-83

### The Bug
```python
# BUGGY:
p.data.add_(p.data, alpha=-lr * weight_decay)
p.data.addcdiv_(m, v.sqrt() + eps, value=-alpha_t)
```

### Why It's Wrong
In AdamW (Decoupled Weight Decay), the weight decay should be applied AFTER the Adam update, not before. The correct order matters because:

1. The original buggy code applies weight decay first, then the Adam step
2. This means weight decay is applied to the OLD weights, but then we also include those old weights in the Adam update
3. The result is that weight decay is effectively applied incorrectly

Additionally, the weight decay in AdamW should be: `p = p - lr * weight_decay * p` (equivalent to `p = p * (1 - lr * weight_decay)`), applied AFTER the Adam gradient step.

The buggy code's order: decay → adam step
The correct order: adam step → decay

### The Fix
```python
# CORRECT:
p.data.addcdiv_(m, v.sqrt() + eps, value=-alpha_t)
p.data.add_(p.data, alpha=-lr * weight_decay)
```

### Symptom
- Regularization won't work as expected
- May lead to different convergence behavior than expected
- Can be subtle and hard to detect without careful comparison to reference implementation

---

## Debugging Strategy Tips

1. **Start with shape checks**: Print tensor shapes at key points to catch dimension mismatches
2. **Unit test components**: Test each module (attention, norm, etc.) in isolation
3. **Compare to reference**: Use PyTorch's built-in modules (nn.MultiheadAttention, nn.LayerNorm) as ground truth
4. **Check gradients**: Verify gradients are flowing and have reasonable magnitudes
5. **Visualize attention**: Plot attention patterns to verify causal masking is working
6. **Loss sanity check**: Loss should be positive and decreasing; negative loss is a red flag
7. **Use small examples**: Test with tiny models (d_model=4, seq_len=3) where you can compute expected values by hand

## Files Modified

- `transformer/core.py`: Bugs 1, 4, 5
- `transformer/attention.py`: Bugs 2, 3
- `transformer/transformer.py`: Bug 6
- `training/loss.py`: Bug 7
- `training/optimizer.py`: Bug 8

