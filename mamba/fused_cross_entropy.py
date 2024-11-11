"""This is a fused cross-entropy and linear layer. Idea is copied 
from https://github.com/linkedin/Liger-Kernel who just copied it from
https://github.com/mgmalek/efficient_cross_entropy
"""

import torch
from torch.autograd import Function
from torch.nn import functional as F


class CrossEntropyLoopedFused(Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, act: torch.Tensor, labels: torch.Tensor):
        bs = act.shape[0]
        weight_grad = torch.zeros_like(weight)
        act_grad = torch.empty_like(act)
        out_loss = torch.tensor(0.0, device=act.device)
        chunksize = 2048

        for b in range(0, bs, chunksize):
            end_idx = min(b + chunksize, bs)

            # Get current batch chunks
            act_chunk = act[b:end_idx]  # [chunk_size, H]
            labels_chunk = labels[b:end_idx]  # [chunk_size]

            # Compute logits
            logits = F.linear(act_chunk, weight)  # [chunk_size, V]

            # Compute softmax and loss
            max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
            exp_logits = torch.exp(logits - max_logits)
            sum_exp = torch.sum(exp_logits, dim=-1, keepdim=True)
            probs = exp_logits / sum_exp  # [chunk_size, V]

            # Compute loss using gather
            correct_logits = torch.gather(
                logits, 1, labels_chunk.unsqueeze(1)
            )  # [chunk_size, 1]
            out_loss += torch.sum(
                max_logits.squeeze()
                + torch.log(sum_exp.squeeze())
                - correct_logits.squeeze()
            )

            # Compute gradients
            dprobs = probs.clone()  # [chunk_size, V]
            dprobs.scatter_(
                1,
                labels_chunk.unsqueeze(1),
                dprobs.gather(1, labels_chunk.unsqueeze(1)) - 1,
            )

            # Accumulate gradients
            weight_grad += dprobs.T @ act_chunk  # [H, V]
            act_grad[b:end_idx] = dprobs @ weight  # [chunk_size, H]

        # Scale gradients
        scale = 1.0 / bs
        weight_grad *= scale
        act_grad *= scale

        ctx.save_for_backward(weight_grad, act_grad)
        return scale * out_loss

    @staticmethod
    def backward(ctx, grad_output):

        (
            weight_grad,
            act_grad,
        ) = ctx.saved_tensors
        return grad_output * weight_grad, grad_output * act_grad, None


@torch.compile
def fused_cross_entropy(lm_head_weight, act, labels):
    return CrossEntropyLoopedFused.apply(lm_head_weight, act, labels)
