import math
from collections.abc import Callable, Iterable

import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m = state["m"]
                v = state["v"]
                state["t"] = state.get("t", 0) + 1
                t = state["t"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                p.data.addcdiv_(m, v.sqrt() + eps, value=-alpha_t)
                p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss


def cosine_lr_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    elif it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (
            1 + math.cos((it - warmup_iters) * math.pi / (cosine_cycle_iters - warmup_iters))
        ) * (max_learning_rate - min_learning_rate)
    return min_learning_rate


def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    eps = 1e-6

    grads = [p.grad for p in parameters if p.grad is not None]

    if len(grads) == 0:
        return torch.tensor(0.0)

    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in grads]), 2)

    if total_norm > max_l2_norm:
        for g in grads:
            g.detach().mul_(max_l2_norm / (total_norm + eps))

    return total_norm


if __name__ == "__main__":
    lrs = [
        1,
        1e1,
        1e2,
        1e3,
    ]

    for lr in lrs:
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=lr)

        print(f"Learning rate: {lr}")
        for i in range(10):
            opt.zero_grad()
            loss = (weights**2).mean()
            print(f"Iter {i}: {loss.cpu().item()}")
            loss.backward()
            opt.step()
        print("-" * 10)
