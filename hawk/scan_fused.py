import warnings
from typing import Tuple

import torch
import triton
import triton.language as tl
from torch.autograd import Function


@triton.jit
def clipped_sqrt_grad(x, MAX_SQRT_GRADIENT=1000):
    min_value = 1.0 / (MAX_SQRT_GRADIENT * MAX_SQRT_GRADIENT)
    scaled_x = 4.0 * x
    clipped_value = tl.where(scaled_x > min_value, scaled_x, min_value)
    return 1.0 / tl.sqrt(clipped_value)


@triton.jit
def softplus_fwd(x):
    return tl.log(1 + tl.exp(x))


@triton.jit
def softplus_bwd(x):
    return tl.exp(x) / (1 + tl.exp(x))


@triton.jit
def sigmoid_bwd(x):
    return tl.sigmoid(x) * (1 - tl.sigmoid(x))


# fmt: off
@triton.jit
def sequential_scan_fwd_kernel(
    x_ptr, x_rg_lru_ptr, a_rg_lru_ptr, a_param_ptr,
    hidden_ptr,
    bs_stride,sq_stride,
    num_context: tl.constexpr,
    numel: tl.constexpr,
    BLOCKSIZE: tl.constexpr
):
    #fmt: on

    C: tl.constexpr = -8.0 

    bs_pid = tl.program_id(0)

    channel_pid = tl.program_id(1)

    offs = tl.arange(0, BLOCKSIZE)
    mask = offs < numel

    x_ptr += bs_pid * bs_stride + (channel_pid*BLOCKSIZE)
    x_rg_lru_ptr += bs_pid * bs_stride + (channel_pid*BLOCKSIZE)
    a_rg_lru_ptr += bs_pid * bs_stride + (channel_pid*BLOCKSIZE)
    hidden_ptr += bs_pid * bs_stride + (channel_pid*BLOCKSIZE)

    a_param = tl.load(a_param_ptr + (channel_pid*BLOCKSIZE) + offs, mask = mask).to(tl.float32)

    # compute first hidden state

    x = tl.load(x_ptr + offs, mask = mask).to(tl.float32)
    x_rg_lru = tl.load(x_rg_lru_ptr + offs, mask = mask).to(tl.float32)
    a_rg_lru = tl.load(a_rg_lru_ptr + offs, mask = mask).to(tl.float32)

    x_rg_lru = tl.sigmoid(x_rg_lru)
    a_rg_lru = tl.sigmoid(a_rg_lru)

    log_a = C * a_rg_lru * softplus_fwd(a_param)
    a_square = tl.exp(2 * log_a)

    multiplier = tl.sqrt(1 - a_square)

    gated_x = x * x_rg_lru

    beta_t = gated_x * multiplier

    # compute h_0 outside loop

    hidden_t = beta_t

    tl.store(hidden_ptr + offs, hidden_t.to(tl.bfloat16), mask = mask)

    for i in range(1, num_context):
        hidden_ptr += sq_stride
        x_ptr += sq_stride
        x_rg_lru_ptr += sq_stride
        a_rg_lru_ptr += sq_stride

        x = tl.load(x_ptr + offs, mask = mask).to(tl.float32)
        x_rg_lru = tl.load(x_rg_lru_ptr + offs, mask = mask).to(tl.float32)
        a_rg_lru = tl.load(a_rg_lru_ptr + offs, mask = mask).to(tl.float32)

        x_rg_lru = tl.sigmoid(x_rg_lru)
        a_rg_lru = tl.sigmoid(a_rg_lru)

        log_a = C * a_rg_lru * softplus_fwd(a_param)
        alpha_t = tl.exp(log_a)

        a_square = tl.exp(2 * log_a)

        multiplier = tl.sqrt(1 - a_square)

        gated_x = x * x_rg_lru

        beta_t = gated_x * multiplier

        hidden_t = (alpha_t * hidden_t) + beta_t

        tl.store(hidden_ptr + offs, hidden_t.to(tl.bfloat16), mask = mask)

#fmt: off
@triton.jit
def sequential_scan_bwd_kernel(
    x_ptr, x_rg_lru_ptr, a_rg_lru_ptr, a_param_ptr, h_saved_ptr,
    d_out_ptr,
    dx_ptr, dx_rg_lru_ptr, da_rg_lru_ptr, da_param_ptr,
    bs_stride, sq_stride,
    aparam_bs_stride,
    num_context: tl.constexpr,
    numel: tl.constexpr,
    BLOCKSIZE: tl.constexpr
):
    
    C: tl.constexpr = -8.0 

    #fmt: on
    bs_pid = tl.program_id(0)

    channel_pid = tl.program_id(1)

    a_param_batched_grad = tl.zeros([BLOCKSIZE], dtype = tl.float32)
    

    h_saved_ptr += (bs_pid * bs_stride) + ((num_context -2)*sq_stride) + (channel_pid*BLOCKSIZE)

    # offset ptrs to correct batch start 
    d_out_ptr += (bs_pid * bs_stride) + ((num_context-1)*sq_stride) + (channel_pid*BLOCKSIZE)

    dx_ptr += (bs_pid * bs_stride) + ((num_context-1)*sq_stride) + (channel_pid*BLOCKSIZE)
    dx_rg_lru_ptr += (bs_pid * bs_stride) + ((num_context-1)*sq_stride) + (channel_pid*BLOCKSIZE)
    da_rg_lru_ptr += (bs_pid * bs_stride) + ((num_context-1)*sq_stride) + (channel_pid*BLOCKSIZE)
    da_param_ptr += (bs_pid * aparam_bs_stride) + (channel_pid*BLOCKSIZE)


    x_ptr += (bs_pid * bs_stride) + ((num_context-1)*sq_stride) + (channel_pid*BLOCKSIZE)
    x_rg_lru_ptr += (bs_pid * bs_stride) + ((num_context-1)*sq_stride) + (channel_pid*BLOCKSIZE)
    a_rg_lru_ptr += (bs_pid * bs_stride) + ((num_context-1)*sq_stride) + (channel_pid*BLOCKSIZE)

    offs = tl.arange(0, BLOCKSIZE)
    
    mask = offs < numel

    a_param = tl.load(a_param_ptr + (channel_pid*BLOCKSIZE) + offs, mask = mask).to(tl.float32)

    # compute (t = T) outside loop
    h_grad = tl.load(d_out_ptr + offs, mask=mask).to(tl.float32)
    h_rec = tl.load(h_saved_ptr + offs, mask=mask).to(tl.float32)

    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    x_rg_lru = tl.load(x_rg_lru_ptr + offs, mask=mask).to(tl.float32)
    a_rg_lru = tl.load(a_rg_lru_ptr + offs, mask=mask).to(tl.float32)

    d_alpha = h_grad*h_rec
    d_beta = h_grad

    log_a = (
            C
            * tl.sigmoid(a_rg_lru)
            * softplus_fwd(a_param)
        )

    a_square_T = tl.exp(2 * log_a)

    i_T = tl.sigmoid(x_rg_lru)

    multiplier_T = tl.sqrt(1 - a_square_T)

    dlog_a = d_alpha * tl.exp(log_a)

    sqrt_grad = clipped_sqrt_grad(1 - a_square_T)
    extra_term = -2.0 * a_square_T

    dlog_a += (
        d_beta
        * (i_T * x)
        *sqrt_grad * extra_term
    )

    a_rg_lru_grad = sigmoid_bwd(a_rg_lru) * dlog_a * C * softplus_fwd(a_param)

    x_grad = d_beta * multiplier_T * i_T

    x_rg_lru_grad = d_beta * multiplier_T * x * sigmoid_bwd(x_rg_lru)

    a_param_batched_grad += dlog_a * C * tl.sigmoid(a_rg_lru) * softplus_bwd(a_param)

    tl.store(dx_ptr + offs, x_grad.to(tl.bfloat16), mask = mask)
    tl.store(dx_rg_lru_ptr+ offs, x_rg_lru_grad.to(tl.bfloat16), mask = mask)
    tl.store(da_rg_lru_ptr+ offs, a_rg_lru_grad.to(tl.bfloat16), mask = mask)


    for _ in range(2, num_context):
        # reduce pointer offsets
        dx_ptr -= sq_stride
        dx_rg_lru_ptr -= sq_stride
        da_rg_lru_ptr -= sq_stride

        x_ptr -= sq_stride
        x_rg_lru_ptr -= sq_stride
        a_rg_lru_ptr -= sq_stride


        h_saved_ptr -= sq_stride
        d_out_ptr -= sq_stride

        a = tl.exp(log_a)

        h_grad = a * h_grad

        grad_out = tl.load(d_out_ptr + offs,mask=mask).to(tl.float32)
        h_rec = tl.load(h_saved_ptr + offs, mask=mask).to(tl.float32)
        h_grad += grad_out

        d_alpha = h_grad * h_rec
        d_beta = h_grad

        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        x_rg_lru = tl.load(x_rg_lru_ptr + offs, mask=mask).to(tl.float32)
        a_rg_lru = tl.load(a_rg_lru_ptr + offs, mask=mask).to(tl.float32)

        log_a = (
            C
            * tl.sigmoid(a_rg_lru)
            * softplus_fwd(a_param)
        )

        a_square_t = tl.exp(2 * log_a)

        i_t = tl.sigmoid(x_rg_lru)

        multiplier_t = tl.sqrt(1 - a_square_t)

        x_grad = d_beta * multiplier_t * i_t

        x_rg_lru_grad = d_beta * multiplier_t * x * sigmoid_bwd(x_rg_lru)

        dlog_a = d_alpha * tl.exp(log_a)
        sqrt_grad = clipped_sqrt_grad(1 - a_square_t)
        extra_term = -2.0 * a_square_t
        
        dlog_a += (
            d_beta
            * (i_t * x)
            *sqrt_grad * extra_term
        )

        a_rg_lru_grad = sigmoid_bwd(a_rg_lru) * dlog_a * C * softplus_fwd(a_param)

        a_param_batched_grad += dlog_a * C * tl.sigmoid(a_rg_lru) * softplus_bwd(a_param)

        tl.store(dx_ptr + offs, x_grad.to(tl.bfloat16), mask = mask)
        tl.store(dx_rg_lru_ptr + offs, x_rg_lru_grad.to(tl.bfloat16), mask = mask)
        tl.store(da_rg_lru_ptr + offs, a_rg_lru_grad.to(tl.bfloat16), mask = mask)

    dx_ptr -= sq_stride
    dx_rg_lru_ptr -= sq_stride
    da_rg_lru_ptr -= sq_stride

    x_ptr -= sq_stride
    x_rg_lru_ptr -= sq_stride
    a_rg_lru_ptr -= sq_stride


    d_out_ptr -= sq_stride

    a = tl.exp(log_a)
    h_grad = a * h_grad
    grad_out = tl.load(d_out_ptr + offs,mask=mask).to(tl.float32)
    h_grad += grad_out

    d_beta = h_grad

    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    x_rg_lru = tl.load(x_rg_lru_ptr + offs, mask=mask).to(tl.float32)
    a_rg_lru = tl.load(a_rg_lru_ptr + offs, mask=mask).to(tl.float32)

    log_a = (
        C
        * tl.sigmoid(a_rg_lru)
        * softplus_fwd(a_param)
    )

    a_square_t = tl.exp(2 * log_a)

    i_t = tl.sigmoid(x_rg_lru)

    multiplier_t = tl.sqrt(1 - a_square_t)

    x_grad = d_beta * multiplier_t * i_t

    x_rg_lru_grad = d_beta * multiplier_t * x * sigmoid_bwd(x_rg_lru)

    sqrt_grad = clipped_sqrt_grad(1 - a_square_t)
    extra_term = -2.0 * a_square_t
    
    dlog_a = (
        d_beta
        * (i_t * x)
        *sqrt_grad * extra_term
    )

    a_rg_lru_grad = sigmoid_bwd(a_rg_lru) * dlog_a * C * softplus_fwd(a_param)

    a_param_batched_grad += dlog_a * C * tl.sigmoid(a_rg_lru) * softplus_bwd(a_param)

    tl.store(dx_ptr + offs, x_grad.to(tl.bfloat16), mask = mask)
    tl.store(dx_rg_lru_ptr + offs, x_rg_lru_grad.to(tl.bfloat16), mask = mask)
    tl.store(da_rg_lru_ptr + offs, a_rg_lru_grad.to(tl.bfloat16), mask = mask)

    tl.store(da_param_ptr + offs,a_param_batched_grad.to(tl.float32), mask = mask)

def sequential_scan_forward(
    x: torch.Tensor,
    x_rg_lru: torch.Tensor,
    a_rg_lru: torch.Tensor,  # [b,sq,d]
    a_param: torch.Tensor,  # [d]
) -> torch.Tensor:
    """Computes forward pass of a linear scan."""

    hidden = torch.empty_like(x)

    b, sq, d = x.shape

    BLOCKSIZE = 64

    num_blocks = d // BLOCKSIZE

    grid = (b,num_blocks)

    warps = 1

    #fmt: off
    sequential_scan_fwd_kernel[grid](
        x,x_rg_lru,a_rg_lru, a_param,
        hidden,
        x.stride(0),x.stride(1),
        sq,d,BLOCKSIZE, 
        num_warps = warps, # type: ignore
        num_stages = 2 # type: ignore
    )
    #fmt: on
    return hidden


def sequential_scan_backward(
    x_saved, # [b,sq,d]
    x_rg_lru_saved, # [b,sq,d]
    a_rg_lru_saved, # [b,sq,d]
    a_param_saved, # [d]
    h_saved, # [b,sq,d]
    grad_out: torch.Tensor,  # [b,sq,d]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes backward pass of a linear scan."""
    
    b,l,d = grad_out.shape

    x_grad = torch.empty_like(x_saved)
    x_rg_lru_grad = torch.empty_like(x_saved)
    a_rg_lru_grad = torch.empty_like(x_saved)
    a_param_batched = torch.empty((b, d), device=x_saved.device, dtype = a_param_saved.dtype)
    
    b, sq, d = h_saved.shape

    BLOCKSIZE = 64

    num_blocks = d // BLOCKSIZE

    grid = (b,num_blocks)

    warps = 1

    # semi-cryptic errors if tensors not contiguous
    assert x_saved.is_contiguous()
    assert x_rg_lru_saved.is_contiguous()
    assert a_rg_lru_saved.is_contiguous()
    assert a_param_saved.is_contiguous()
    assert h_saved.is_contiguous()

    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()
        warnings.warn("`grad_out` tensor is not contiguous. Setting to contiguous and attempting to continue. This may impact runtime.")
    assert grad_out.is_contiguous()

    #fmt: off
    sequential_scan_bwd_kernel[grid](
        x_saved, x_rg_lru_saved, a_rg_lru_saved, a_param_saved, h_saved,
        grad_out, 
        x_grad, x_rg_lru_grad, a_rg_lru_grad, a_param_batched,
        x_saved.stride(0), x_saved.stride(1),
        a_param_batched.stride(0),
        sq, d, BLOCKSIZE, 
        num_warps = warps, # type: ignore
        num_stages = 2 # type: ignore
    )
    #fmt: on

    return x_grad, x_rg_lru_grad, a_rg_lru_grad, a_param_batched.sum(dim = 0)

class DiagonalRecurrence(Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type = 'cuda', cast_inputs=torch.bfloat16) # type: ignore
    def forward(ctx, x,x_rg_lru,a_rg_lru,a_param) -> torch.Tensor:
        h = sequential_scan_forward(x,x_rg_lru,a_rg_lru,a_param)
        ctx.save_for_backward(x,x_rg_lru,a_rg_lru,a_param,h)

        return h

    @staticmethod
    @torch.amp.custom_bwd(device_type = 'cuda') # type: ignore
    def backward(ctx, grad_output) -> Tuple[torch.Tensor, torch.Tensor, None]: # type: ignore
        (x,x_rg_lru,a_rg_lru,a_param,h_saved) = ctx.saved_tensors

        x_grad, x_rg_lru_grad, a_rg_lru_grad, a_param_grad = sequential_scan_backward(x,x_rg_lru,a_rg_lru,a_param, h_saved, grad_output)

        return x_grad, x_rg_lru_grad, a_rg_lru_grad, a_param_grad


fused_linear_scan = DiagonalRecurrence.apply
