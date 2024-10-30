from typing import Tuple

import torch
import triton
import triton.language as tl
from torch.autograd import Function


# from https://srush.github.io/annotated-mamba/hard.html
@triton.jit
def rol(a1, b1_last, b1_cur, a2, b2_last, b2_cur):
    return a1 + a2, tl.where(a2 == 1, b1_cur, 0) + b2_last, b2_cur


@triton.jit
def roll(y, dim, rev=0):
    _, rh2, _ = tl.associative_scan((1 + 0 * y, 0.0 * y, y), dim, rol, reverse=rev)
    return rh2


@triton.jit
def associative_binary_op(fl, xl, fr, xr):
    f = fr * fl
    x = fr * xl + xr
    return f, x


@triton.jit
def softplus_fwd(x):
    return tl.log(1 + tl.exp(x))


@triton.jit
def softplus_bwd(x):
    return tl.exp(x) / (1 + tl.exp(x))


@triton.jit
def parallel_scan_fwd_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    delta_ptr,
    x_ptr,
    o_ptr,
    A_row_stride,
    A_col_stride,
    B_b_stride,
    B_n_stride,
    x_b_stride,
    x_d_stride,
    BLOCKSIZE_L: tl.constexpr,
    N: tl.constexpr,
):

    # parallelize over (batch, channel) dimension
    b_pid = tl.program_id(0)
    d_pid = tl.program_id(1)

    x_ptr += b_pid * x_b_stride + d_pid * x_d_stride
    delta_ptr += b_pid * x_b_stride + d_pid * x_d_stride

    l_offs = tl.arange(0, BLOCKSIZE_L)

    C_ptr += b_pid * B_b_stride
    B_ptr += b_pid * B_b_stride

    x = tl.load(x_ptr + l_offs).to(tl.float32)  # [L]
    delta = softplus_fwd(tl.load(delta_ptr + l_offs).to(tl.float32))  # [L]

    A_ptr += A_row_stride * d_pid

    y = tl.zeros(
        [
            BLOCKSIZE_L,
        ],
        dtype=tl.float32,
    )

    # backward pass needs to then reduce something somehow?
    for chan_idx in range(N):

        B_ch = tl.load(B_ptr + chan_idx * B_n_stride + l_offs).to(tl.float32)

        C_ch = tl.load(C_ptr + chan_idx * B_n_stride + l_offs).to(tl.float32)

        deltaB_ch = delta * x * B_ch  # [L]

        A_ch = tl.load(A_ptr + chan_idx * A_col_stride).to(tl.float32)  # single value

        deltaA_ch = tl.exp(A_ch * delta)  # [L]

        _, o = tl.associative_scan(
            (deltaA_ch, deltaB_ch), axis=0, combine_fn=associative_binary_op
        )

        y += o * C_ch

    o_ptr += b_pid * x_b_stride + d_pid * x_d_stride
    tl.store(o_ptr + l_offs, value=y.to(tl.bfloat16))


def parallel_scan_forward(
    A: torch.Tensor,  # [d_in, n]
    delta: torch.Tensor,  # [b,d,l]
    B: torch.Tensor,  # [b,n,l]
    C: torch.Tensor,  # [b,n,l]
    x: torch.Tensor,  # [b,d,l]
) -> torch.Tensor:  # [b,d,l]

    n = A.shape[1]
    b, d, l = x.shape

    o = torch.empty_like(x)

    BLOCKSIZE_L = l
    N = n

    grid = (b, d)

    num_warps = 2
    if BLOCKSIZE_L > 1024:
        num_warps = 4

    # fmt: off
    parallel_scan_fwd_kernel[grid](
        A,B,C,delta,
        x,o,
        A.stride(0),A.stride(1),
        B.stride(0),B.stride(1),
        x.stride(0),x.stride(1),
        BLOCKSIZE_L,N,
        enable_fp_fusion = True, 
        num_warps = num_warps
    )
    # fmt: on

    return o


# fmt: off
@triton.jit
def parallel_scan_bwd_kernel(
    A_ptr,B_ptr,C_ptr,delta_ptr,
    x_ptr,dout_ptr,
    dA_ptr,dB_ptr,dC_ptr,ddelta_ptr, 
    dx_ptr, delta_shifted_ptr,
    A_row_stride,A_col_stride,
    dA_b_stride,dA_row_stride,dA_col_stride,
    B_b_stride,B_n_stride,
    dB_b_stride,dB_d_stride, dB_n_stride,
    x_b_stride, x_d_stride,
    BLOCKSIZE_L: tl.constexpr, # sequence length
    N: tl.constexpr, # state size
):
    # fmt: on

    # parallelize over (batch, channel) dimension
    b_pid = tl.program_id(0)
    d_pid = tl.program_id(1)

    dB_ptr += (b_pid * dB_b_stride) + d_pid * dB_d_stride
    dC_ptr += (b_pid * dB_b_stride) + d_pid * dB_d_stride
    dA_ptr += (b_pid*dA_b_stride) + d_pid*dA_row_stride

    x_ptr += b_pid * x_b_stride + d_pid * x_d_stride
    dx_ptr += b_pid * x_b_stride + d_pid * x_d_stride
    ddelta_ptr += b_pid * x_b_stride + d_pid * x_d_stride
    delta_ptr += b_pid * x_b_stride + d_pid * x_d_stride


    delta_shifted_ptr += b_pid * x_b_stride + d_pid * x_d_stride
    
    dout_ptr += b_pid * x_b_stride + d_pid * x_d_stride

    l_offs = tl.arange(0, BLOCKSIZE_L)

    x = tl.load(x_ptr + l_offs).to(tl.float32)  # [L]
    delta = softplus_fwd(tl.load(delta_ptr + l_offs).to(tl.float32))  # [L]

    delta_shifted = softplus_fwd(tl.load(delta_shifted_ptr + l_offs).to(tl.float32))  # [L]

    A_ptr += A_row_stride * d_pid

    C_ptr += b_pid * B_b_stride
    B_ptr += b_pid * B_b_stride
    

    dX = tl.zeros(
        [
            BLOCKSIZE_L,
        ],
        dtype=tl.float32,
    )


    ddelta = tl.zeros(
        [
            BLOCKSIZE_L,
        ],
        dtype=tl.float32,
    )

    for chan_idx in range(N):

        B_ch = tl.load(B_ptr + chan_idx * B_n_stride + l_offs).to(tl.float32)
        deltaB_ch = delta * x * B_ch  # [L]
        A_ch = tl.load(A_ptr + chan_idx * A_col_stride).to(tl.float32)  # single value
        deltaA_ch = tl.exp(A_ch * delta)  # [L]

        dout = tl.load(dout_ptr + l_offs).to(tl.float32)
        C = tl.load(C_ptr + chan_idx * B_n_stride + l_offs).to(tl.float32)

        _, h_ch = tl.associative_scan( 
            (deltaA_ch, deltaB_ch), axis=0, combine_fn=associative_binary_op
        )


        C_ch_grad = dout * h_ch

        tl.store(dC_ptr + (chan_idx*dB_n_stride) + l_offs, value=C_ch_grad.to(tl.bfloat16))

        out_h_grad = dout * C

        # reverse scan

        deltaA_ch_shifted = tl.exp(A_ch * delta_shifted)  # [L]
        deltaA_ch_shifted = tl.where(tl.arange(0, BLOCKSIZE_L) < (BLOCKSIZE_L - 1), deltaA_ch_shifted, 0.0)
        _, h_grad = tl.associative_scan((deltaA_ch_shifted, out_h_grad), axis=0, combine_fn=associative_binary_op, reverse=True)

        dX += (h_grad * delta * B_ch)

        B_ch_grad = h_grad * delta * x

        tl.store(dB_ptr + (chan_idx*dB_n_stride) + l_offs, value=B_ch_grad.to(tl.bfloat16))

        # need a way to create h_tilde from h_ch like: [0,h_0,h_1,...]
        h_ch_rolled = roll(h_ch, dim = 0)

        intermediate = h_grad * h_ch_rolled * deltaA_ch

        ddelta += (intermediate * A_ch)
        ddelta += (h_grad * (B_ch * x))

        # print(intermediate.shape, delta.shape)
        A_batched_grad = tl.sum(intermediate * delta)

        tl.store(dA_ptr + (chan_idx), value=A_batched_grad.to(tl.float32))

    # kinda dumb but w.e.
    delta = tl.load(delta_ptr + l_offs).to(tl.float32)
    ddelta = ddelta * softplus_bwd(delta)

    tl.store(ddelta_ptr + l_offs, value= ddelta.to(tl.bfloat16))
    tl.store(dx_ptr + l_offs, value= dX.to(tl.bfloat16))

def parallel_scan_backward(A, delta, B, C, x, grad_output) -> torch.Tensor:

    n = A.shape[1]
    b, d, l = x.shape
    
    BLOCKSIZE_L = l
    N = n

    grid = (b, d)
    
    # [b, d, n]
    A_batch_grad = torch.empty((b, A.shape[0], A.shape[1]), device=x.device)
    

    B_grad = torch.empty((b, d, n, l), device=x.device, dtype = torch.bfloat16) # [b, d, n, l] 
    C_grad = torch.empty((b, d, n, l), device=x.device, dtype = torch.bfloat16) # [b, d, n, l] 

    delta_grad = torch.empty_like(delta) # [b,d,l]
    x_grad = torch.empty_like(x) # [b,d,l]
    
    num_warps = 2
    if BLOCKSIZE_L > 1024:
        num_warps = 4 
    
    delta_shifted = torch.cat([delta, torch.ones_like(delta[:, :, :1])], dim=-1)[:, :, 1:].contiguous()
    parallel_scan_bwd_kernel[grid](
        A, B, C, delta, x, grad_output,
        A_batch_grad, B_grad, C_grad, delta_grad, x_grad, delta_shifted,
        A.stride(0), A.stride(1),
        A_batch_grad.stride(0), A_batch_grad.stride(1), A_batch_grad.stride(2),
        B.stride(0),B.stride(1), 
        B_grad.stride(0),B_grad.stride(1),B_grad.stride(2), 
        x.stride(0), x.stride(1),
        BLOCKSIZE_L,N,
        enable_fp_fusion = True,
        num_warps = num_warps
    )

    return A_batch_grad.sum(dim=0), B_grad.sum(dim = 1), C_grad.sum(dim = 1), delta_grad, x_grad


class MambaRecurrence(Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)  # type: ignore
    def forward(ctx, A, delta, B, C, x) -> torch.Tensor:

        if not A.is_contiguous():
            A = A.contiguous()

        if not C.is_contiguous():
            C = C.contiguous()

        if not delta.is_contiguous():
            delta = delta.contiguous()

        if not B.is_contiguous():
            B = B.contiguous()

        if not x.is_contiguous():
            x = x.contiguous()

        o = parallel_scan_forward(A, delta, B, C, x)

        ctx.save_for_backward(A, delta, B, C, x)

        return o

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")  # type: ignore
    def backward(ctx, grad_output) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore
        (A, delta, B, C, x) = ctx.saved_tensors

        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        A_batch_grad, B_grad, C_grad, delta_grad, x_grad = parallel_scan_backward(A, delta, B, C, x, grad_output)

        return A_batch_grad.to(A.dtype), delta_grad, B_grad, C_grad, x_grad

fused_scan = MambaRecurrence.apply
