import warnings
from typing import Tuple

import torch
import triton
import triton.language as tl
from torch.autograd import Function


# fmt: off
@triton.jit
def sequential_scan_fwd_kernel(
    alpha_ptr,beta_ptr,hidden_ptr,
    bs_stride,sq_stride,
    num_context: tl.constexpr,
    numel: tl.constexpr,
    BLOCKSIZE: tl.constexpr
):
    #fmt: on

    bs_pid = tl.program_id(0)

    alpha_ptr += bs_pid * bs_stride
    beta_ptr += bs_pid * bs_stride
    hidden_ptr += bs_pid * bs_stride

    offs = tl.arange(0, BLOCKSIZE)
    mask = offs < numel
    # compute h_0 outside loop

    hidden_t = tl.load(beta_ptr + offs, mask = mask).to(tl.float32)

    tl.store(hidden_ptr + offs, hidden_t.to(tl.bfloat16), mask = mask)

    for i in range(1, num_context):
        beta_ptr += sq_stride
        alpha_ptr += sq_stride
        hidden_ptr += sq_stride

        alpha_t = tl.load(alpha_ptr + offs, mask = mask).to(tl.float32)
        beta_t = tl.load(beta_ptr + offs, mask = mask).to(tl.float32)

        hidden_t = (alpha_t * hidden_t) + beta_t

        tl.store(hidden_ptr + offs, hidden_t.to(tl.bfloat16), mask = mask)

#fmt: off
@triton.jit
def sequential_scan_bwd_kernel(
    alpha_saved_ptr,h_saved_ptr,d_out_ptr,
    d_alpha_ptr,d_beta_ptr, 
    bs_stride, sq_stride,
    num_context: tl.constexpr,
    numel: tl.constexpr,
    BLOCKSIZE: tl.constexpr
):
    #fmt: on
    bs_pid = tl.program_id(0)
    

    # offset ptrs to correct batch start 
    alpha_saved_ptr += (bs_pid * bs_stride) + ((num_context)*sq_stride)
    h_saved_ptr += (bs_pid * bs_stride) + ((num_context -2)*sq_stride) 
    d_out_ptr += (bs_pid * bs_stride) + ((num_context-1)*sq_stride)

    d_alpha_ptr += (bs_pid * bs_stride) + ((num_context-1)*sq_stride)
    d_beta_ptr += (bs_pid * bs_stride) + ((num_context-1)*sq_stride)

    offs = tl.arange(0, BLOCKSIZE)
    
    mask = offs < numel

    # compute (t = T) outside loop
    h_grad = tl.load(d_out_ptr + offs, mask=mask).to(tl.float32)
    h_rec = tl.load(h_saved_ptr + offs, mask=mask).to(tl.float32)

    d_alpha = h_grad*h_rec
    d_beta = h_grad

    tl.store(d_alpha_ptr + offs, d_alpha.to(tl.bfloat16),mask=mask)
    tl.store(d_beta_ptr + offs, d_beta.to(tl.bfloat16),mask=mask)
    
    for _ in range(2, num_context):
        # reduce pointer offsets
        d_alpha_ptr -= sq_stride
        d_beta_ptr -= sq_stride
        h_saved_ptr -= sq_stride
        d_out_ptr -= sq_stride
        alpha_saved_ptr -= sq_stride

        alpha = tl.load(alpha_saved_ptr + offs,mask=mask).to(tl.float32)
        grad_out = tl.load(d_out_ptr + offs,mask=mask).to(tl.float32)
        h_rec = tl.load(h_saved_ptr + offs,mask=mask).to(tl.float32)


        h_grad = alpha * h_grad
        h_grad += grad_out
        
        d_alpha = h_grad * h_rec
        d_beta = h_grad

        tl.store(d_alpha_ptr + offs, d_alpha.to(tl.bfloat16),mask=mask)
        tl.store(d_beta_ptr + offs, d_beta.to(tl.bfloat16),mask=mask)


    # first grad (t = 0)
    d_alpha_ptr -= sq_stride
    d_beta_ptr -= sq_stride
    d_out_ptr -= sq_stride
    alpha_saved_ptr -= sq_stride

    alpha = tl.load(alpha_saved_ptr + offs,mask=mask).to(tl.float32)
    grad_out = tl.load(d_out_ptr + offs,mask=mask).to(tl.float32)
    
    h_grad = alpha * h_grad
    h_grad += grad_out
    d_beta = h_grad

    d_alpha = tl.zeros_like(d_beta).to(tl.float32)

    tl.store(d_alpha_ptr + offs, d_alpha.to(tl.bfloat16),mask=mask)
    tl.store(d_beta_ptr + offs, d_beta.to(tl.bfloat16),mask=mask)

def sequential_scan_forward(
    alpha: torch.Tensor,  # [b,sq,d]
    beta: torch.Tensor,  # [b,sq,d]
) -> torch.Tensor:
    """Computes forward pass of a linear scan."""

    hidden = torch.empty_like(alpha)

    b, sq, d = alpha.shape

    BLOCKSIZE = triton.next_power_of_2(d)

    grid = (b,)

    match d:
        case _ if d <= 256:
            warps = 1

        case _ if d <= 512:
            warps = 2

        case _ if d <= 1024:
            warps = 4

        case _ :
            warps = 8


    #fmt: off
    sequential_scan_fwd_kernel[grid](
        alpha,beta,hidden,
        alpha.stride(0),alpha.stride(1),
        sq,d,BLOCKSIZE, 
        num_warps = warps, # type: ignore
        num_stages = 2 # type: ignore
    )
    #fmt: on
    return hidden


def sequential_scan_backward(
    alpha_saved: torch.Tensor,  # [b,sq,d]
    h_saved: torch.Tensor,  # [b,sq,d]
    grad_out: torch.Tensor,  # [b,sq,d]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes backward pass of a linear scan."""

    alpha_grad = torch.empty_like(alpha_saved)
    beta_grad = torch.empty_like(alpha_saved)
    
    b, sq, d = alpha_saved.shape

    BLOCKSIZE = triton.next_power_of_2(d)

    grid = (b,)

    match d:
        case _ if d <= 256:
            warps = 1

        case _ if d <= 512:
            warps = 2

        case _ if d <= 1024:
            warps = 4

        case _ :
            warps = 8

    # semi-cryptic errors if tensors not contiguous
    assert alpha_saved.is_contiguous()
    assert h_saved.is_contiguous()

    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()
        warnings.warn("`grad_out` tensor is not contiguous. Setting to contiguous and attempting to continue. This may impact runtime.")
    assert grad_out.is_contiguous()
 
    #fmt: off
    sequential_scan_bwd_kernel[grid](
        alpha_saved,h_saved, grad_out,
        alpha_grad, beta_grad,
        alpha_saved.stride(0), alpha_saved.stride(1),
        sq, d, BLOCKSIZE, 
        num_warps = warps, # type: ignore
        num_stages = 2 # type: ignore
    )
    #fmt: on

    return alpha_grad, beta_grad

class DiagonalRecurrence(Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type = 'cuda', cast_inputs=torch.bfloat16) # type: ignore
    def forward(ctx, input_alpha, input_beta) -> torch.Tensor:
        h = sequential_scan_forward(input_alpha, input_beta)
        ctx.save_for_backward(input_alpha, h)

        return h

    @staticmethod
    @torch.amp.custom_bwd(device_type = 'cuda') # type: ignore
    def backward(ctx, grad_output) -> Tuple[torch.Tensor, torch.Tensor, None]: # type: ignore
        (input_alpha, h) = ctx.saved_tensors

        alpha_grad, beta_grad = sequential_scan_backward(input_alpha, h, grad_output)

        return alpha_grad, beta_grad, None


linear_scan = DiagonalRecurrence.apply
