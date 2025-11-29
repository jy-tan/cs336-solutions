# Buggy Transformer - Answer Key

This document contains all the bugs injected into this codebase for debugging practice.
**Total bugs: 7**

---

## Bug 1: Wrong Attention Scaling (Multiply instead of Divide)

**File:** `transformer/attention.py`
**Line:** ~47
**Severity:** Critical - will cause numerical instability and poor training

### Buggy Code
```python
scaled_qk_dot_product = qk_dot_product * (d_k**0.5)
```

### Fixed Code
```python
scaled_qk_dot_product = qk_dot_product / (d_k**0.5)
```

### Explanation
The attention mechanism requires **dividing** by √d_k to prevent the dot products from growing too large as the dimension increases. Multiplying instead of dividing will cause the softmax to become very peaky (approaching one-hot), making gradients vanish and training unstable.

### How to Detect
- Attention weights become nearly one-hot (one position gets ~1.0, others ~0.0)
- Gradients through attention become very small
- Loss doesn't decrease or becomes NaN

---

## Bug 2: Softmax on Wrong Dimension

**File:** `transformer/attention.py`
**Line:** ~50
**Severity:** Critical - fundamentally breaks attention mechanism

### Buggy Code
```python
attention_weights = softmax(scaled_qk_dot_product, dimension=-2)
```

### Fixed Code
```python
attention_weights = softmax(scaled_qk_dot_product, dimension=-1)
```

### Explanation
Softmax should be applied over the **keys** dimension (last dimension, -1), so that for each query, the attention weights over all keys sum to 1. Applying softmax over the queries dimension (-2) instead means each key's weights over all queries sum to 1, which is semantically wrong.

### How to Detect
- Check that `attention_weights.sum(dim=-1)` equals 1.0 for each query position
- If softmax is on wrong dim, `attention_weights.sum(dim=-2)` will equal 1.0 instead
- Model will produce nonsensical outputs

---

## Bug 3: RoPE Applied to Values (Should Only Be Q and K)

**File:** `transformer/attention.py`  
**Line:** ~138
**Severity:** High - corrupts the value representations

### Buggy Code
```python
if token_positions is not None:
    Q = self.rope(Q, token_positions=token_positions)
    K = self.rope(K, token_positions=token_positions)
    V = self.rope(V, token_positions=token_positions)
```

### Fixed Code
```python
if token_positions is not None:
    Q = self.rope(Q, token_positions=token_positions)
    K = self.rope(K, token_positions=token_positions)
```

### Explanation
Rotary Position Embeddings (RoPE) encode positional information in Q and K so that their dot product naturally encodes relative position. The Values (V) should NOT have positional rotation applied - they should preserve the original content representations that get aggregated by the attention weights.

### How to Detect
- Compare with a known-correct implementation
- Values should remain content-based, not position-dependent
- Model may still train but will have degraded performance

---

## Bug 4: Missing Epsilon in RMSNorm

**File:** `transformer/core.py`
**Line:** ~109
**Severity:** Medium - causes NaN/Inf with certain inputs

### Buggy Code
```python
rms_inv = torch.rsqrt(reduce(x**2, "... d_model -> ... 1", "mean"))
```

### Fixed Code
```python
rms_inv = torch.rsqrt(reduce(x**2, "... d_model -> ... 1", "mean") + self.eps)
```

### Explanation
The epsilon term prevents division by zero when the input has very small values. Without it, if `x` is all zeros or very small, `rsqrt(0)` = infinity, causing NaN values to propagate through the network.

### How to Detect
- NaN values appearing during training
- Check for inf/nan after normalization layers
- More likely to occur with certain weight initializations or inputs

---

## Bug 5: Wrong Rotation Signs in RoPE

**File:** `transformer/core.py`
**Line:** ~232-233
**Severity:** High - breaks positional encoding

### Buggy Code
```python
row1 = x1 * cos + x2 * sin
row2 = x1 * sin - x2 * cos
```

### Fixed Code
```python
row1 = x1 * cos - x2 * sin
row2 = x1 * sin + x2 * cos
```

### Explanation
The RoPE rotation matrix should be:
```
[cos θ, -sin θ]
[sin θ,  cos θ]
```
The buggy version has wrong signs, which means the rotation doesn't properly encode positions. The relative position dot product property `R(m)^T R(n) = R(n-m)` will not hold.

### How to Detect
- Compare RoPE output with reference implementation
- Check that rotating by angle θ then -θ returns to original
- Model will fail to learn positional relationships properly

---

## Bug 6: Missing Residual Connection in FFN

**File:** `transformer/transformer.py`
**Line:** ~61-63
**Severity:** Critical - fundamentally breaks transformer architecture

### Buggy Code
```python
# Feed-forward sublayer
x = self.ln2(x)
x = self.ffn(x)
```

### Fixed Code
```python
# Feed-forward sublayer
x_orig = x.clone()
x = self.ln2(x)
x = self.ffn(x) + x_orig
```

### Explanation
Residual connections are essential for training deep networks. They allow gradients to flow directly through the network and help preserve information from earlier layers. Without the residual connection, the model becomes much harder to train and loses the "identity shortcut" that transformers rely on.

### How to Detect
- Gradients vanish in deep models
- Loss doesn't decrease
- Check that output has contribution from input (not just FFN output)

---

## Bug 7: Wrong Cross-Entropy Sign

**File:** `training/loss.py`
**Line:** ~34
**Severity:** Critical - optimizer will maximize loss instead of minimizing

### Buggy Code
```python
loss = target_log_probs.mean()
```

### Fixed Code
```python
loss = -target_log_probs.mean()
```

### Explanation
Cross-entropy loss is the **negative** log probability of the correct class. Log probabilities are always ≤ 0, so without the negative sign, the loss will be negative and the optimizer will try to make it more negative (i.e., maximize log probability of wrong classes).

### How to Detect
- Loss is negative
- Loss decreases toward negative infinity
- Model predictions become random or adversarially wrong

---

## Bug 8: AdamW Weight Decay Applied to Momentum

**File:** `training/optimizer.py`
**Line:** ~83
**Severity:** Medium - subtly wrong optimization behavior

### Buggy Code
```python
p.data.addcdiv_(m, v.sqrt() + eps, value=-alpha_t)
m.add_(p.data, alpha=-lr * weight_decay)
```

### Fixed Code
```python
p.data.addcdiv_(m, v.sqrt() + eps, value=-alpha_t)
p.data.add_(p.data, alpha=-lr * weight_decay)
```

### Explanation
AdamW (decoupled weight decay) should apply weight decay directly to the **parameters** (`p.data`), not to the momentum (`m`). The buggy version corrupts the momentum estimate with weight values, breaking the adaptive learning rate mechanism.

### How to Detect
- Training is unstable or slower than expected
- Compare optimizer state with reference implementation
- Weight decay isn't working as expected (large weights aren't penalized properly)

---

## Summary Table

| # | Bug | File | Severity | Category |
|---|-----|------|----------|----------|
| 1 | Multiply instead of divide in attention scaling | attention.py | Critical | Numerical |
| 2 | Softmax on wrong dimension | attention.py | Critical | Shape/Dimension |
| 3 | RoPE applied to V | attention.py | High | Architecture |
| 4 | Missing epsilon in RMSNorm | core.py | Medium | Numerical |
| 5 | Wrong rotation signs in RoPE | core.py | High | Math/Formula |
| 6 | Missing residual connection | transformer.py | Critical | Architecture |
| 7 | Wrong sign in cross-entropy | loss.py | Critical | Math/Formula |
| 8 | Weight decay on momentum | optimizer.py | Medium | Algorithm |

---

## Debugging Tips

1. **Start with shape checks** - Add print statements for tensor shapes at each step
2. **Check numerical properties** - Are attention weights summing to 1? Are there NaN/Inf values?
3. **Compare with reference** - Use PyTorch's built-in modules (nn.MultiheadAttention, F.cross_entropy) as ground truth
4. **Unit test components** - Test each module in isolation before full forward pass
5. **Check the math** - Verify formulas against papers/references (especially signs and dimensions)

