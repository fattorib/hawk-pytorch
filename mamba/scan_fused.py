from typing import Tuple

import torch
import triton
import triton.language as tl
from torch.autograd import Function


@triton.jit
def softplus_fwd(x):
    return tl.log(1 + tl.exp(x))


@triton.jit
def softplus_bwd(x):
    return tl.exp(x) / (1 + tl.exp(x))


# fmt: off
@triton.jit
def sequential_scan_fwd_kernel(
    A_ptr,B_ptr,delta_ptr,x_ptr,
    hidden_ptr,
    A_r_stride, A_c_stride, 
    B_b_stride, B_l_stride,
    x_b_stride, x_l_stride,
    o_b_stride, o_l_stride,
    num_context: tl.constexpr,
    numel_ld_d: tl.constexpr,
    numel_ld_n: tl.constexpr,
    numel_st: tl.constexpr,
    BLOCKSIZE_LD_N: tl.constexpr,
    BLOCKSIZE_LD_D: tl.constexpr,
    BLOCKSIZE_ST: tl.constexpr
):
    #fmt: on

    bs_pid = tl.program_id(0)

    B_ptr += bs_pid * B_b_stride
    delta_ptr += bs_pid * x_b_stride
    x_ptr += bs_pid * x_b_stride

    hidden_ptr += bs_pid * o_b_stride

    offs_ld_n = tl.arange(0, BLOCKSIZE_LD_N)
    offs_ld_d = tl.arange(0, BLOCKSIZE_LD_D)

    offs_st = tl.arange(0, BLOCKSIZE_ST)[None,:]
    

    delta_i = tl.load(delta_ptr + offs_ld_d,mask = offs_ld_d < numel_ld_d).to(tl.float32)[None,:] # [1, D]
    
    delta_i = softplus_fwd(delta_i)

    x_i = tl.load(x_ptr + offs_ld_d,mask = offs_ld_d < numel_ld_d ).to(tl.float32)[None,:] # [1, D]
    B_i = tl.load(B_ptr + offs_ld_n,mask = offs_ld_n < numel_ld_n ).to(tl.float32)[None,:] # [1, N]

    recc = (delta_i.expand_dims(2) * B_i.expand_dims(1) * x_i.expand_dims(2)).reshape(1, BLOCKSIZE_ST)

    tl.store(hidden_ptr + offs_st, recc.to(tl.bfloat16),mask = offs_st < numel_st)

    A_block_ptr = tl.make_block_ptr(A_ptr, 
                                    shape = (BLOCKSIZE_LD_D, BLOCKSIZE_LD_N), 
                                    block_shape= (BLOCKSIZE_LD_D, BLOCKSIZE_LD_N), 
                                    offsets=(0,0), 
                                    strides = (A_r_stride, A_c_stride), order = (1,0))
    
    A = tl.load(A_block_ptr)[None,:]

    for _ in range(1, num_context):
        B_ptr += B_l_stride
        delta_ptr += x_l_stride
        x_ptr += x_l_stride
        hidden_ptr += o_l_stride

        delta_i = tl.load(delta_ptr + offs_ld_d,mask = offs_ld_d < numel_ld_d).to(tl.float32)[None,:] # [1, D]
        
        delta_i = softplus_fwd(delta_i)
        
        x_i = tl.load(x_ptr + offs_ld_d,mask = offs_ld_d < numel_ld_d ).to(tl.float32)[None,:] # [1, D]
        B_i = tl.load(B_ptr + offs_ld_n,mask = offs_ld_n < numel_ld_n ).to(tl.float32)[None,:] # [1, N]

        deltaB_i = (delta_i.expand_dims(2) * B_i.expand_dims(1) * x_i.expand_dims(2)).reshape(1, BLOCKSIZE_ST)

        deltaA_i = tl.exp(A * delta_i.expand_dims(2)).reshape(1, BLOCKSIZE_ST)

        recc = (deltaA_i*recc) + deltaB_i

        tl.store(hidden_ptr + offs_st, recc.to(tl.bfloat16),mask = offs_st < numel_st)


@triton.jit
def sequential_scan_bwd_kernel(
    A_ptr,B_ptr, delta_ptr, x_ptr, o_ptr,d_out_ptr,
    dA_ptr,dB_ptr,ddelta_ptr, dx_ptr, 
    A_r_stride, A_c_stride,
    dA_bs_stride, dA_row_stride,
    delta_bs_stride, delta_sq_stride,
    B_bs_stride, B_sq_stride,
    o_bs_stride, o_sq_stride,
    num_context: tl.constexpr,
    numel_d: tl.constexpr,
    numel_n: tl.constexpr,
    numel_nd: tl.constexpr,
    BLOCKSIZE_N: tl.constexpr,
    BLOCKSIZE_D: tl.constexpr,
    BLOCKSIZE_ND: tl.constexpr
):
    #fmt: on
    bs_pid = tl.program_id(0)
    

    # offset ptrs to correct batch start 
    A_block_ptr = tl.make_block_ptr(A_ptr, 
                                    shape = (BLOCKSIZE_D, BLOCKSIZE_N), 
                                    block_shape= (BLOCKSIZE_D, BLOCKSIZE_N), 
                                    offsets=(0,0), 
                                    strides = (A_r_stride, A_c_stride), order = (1,0))
    
   
    
    A = tl.load(A_block_ptr) # [BLOCKSIZE_D, BLOCKSIZE_N]

    A_grad = tl.zeros([BLOCKSIZE_D, BLOCKSIZE_N], dtype = tl.float32)
    

    o_ptr += (bs_pid * o_bs_stride) + ((num_context -2)*o_sq_stride) 
    d_out_ptr += (bs_pid * o_bs_stride) + ((num_context-1)*o_sq_stride)
    delta_ptr += (bs_pid * delta_bs_stride) + ((num_context-1)*delta_sq_stride)
    ddelta_ptr += (bs_pid * delta_bs_stride) + ((num_context-1)*delta_sq_stride)
    B_ptr += (bs_pid * B_bs_stride) + ((num_context-1)*B_sq_stride)
    x_ptr += (bs_pid * delta_bs_stride) + ((num_context-1)*delta_sq_stride)
    dx_ptr += (bs_pid * delta_bs_stride) + ((num_context-1)*delta_sq_stride)
    dB_ptr += (bs_pid * B_bs_stride) + ((num_context-1)*B_sq_stride)

    offs_nd = tl.arange(0, BLOCKSIZE_ND)
    offs_n = tl.arange(0, BLOCKSIZE_N)
    offs_d = tl.arange(0, BLOCKSIZE_D)
    
    mask_nd = offs_nd < numel_nd
    mask_n = offs_n < numel_n
    mask_d = offs_d < numel_d

    # compute (t = T) outside loop
    h_grad = tl.load(d_out_ptr + offs_nd, mask=mask_nd).to(tl.float32) # [ND]
    h_rec = tl.load(o_ptr + offs_nd, mask=mask_nd).to(tl.float32) # [ND]

    B = tl.load(B_ptr + offs_n, mask = mask_n).to(tl.float32) # [N]
    x = tl.load(x_ptr + offs_d, mask = mask_d).to(tl.float32) # [D]

    delta = tl.load(delta_ptr + offs_d, mask = mask_d).to(tl.float32) # [D]

    delta_bwd_act = softplus_bwd(delta)
    delta = softplus_fwd(delta)


    intermediate = (h_grad * h_rec).reshape(BLOCKSIZE_D, BLOCKSIZE_N) * tl.exp(delta[:,None] * A) # [D,N] * ([D,1] * [D,N])

    delta_grad = tl.sum(intermediate * A, axis = -1) # [D,]

    # [D,N] * ([1, N] * [D, 1] -> [D,N])
    delta_grad += tl.sum(h_grad.reshape(BLOCKSIZE_D, BLOCKSIZE_N) * (B[None,:] * x[:,None]), axis = -1)
    
    delta_grad = delta_grad*delta_bwd_act
    
    # store delta grad -> this is correct!
    tl.store(ddelta_ptr + offs_d, mask = mask_d, value=delta_grad.to(tl.bfloat16))
    

    # x_grad -> this is correct!
    x_grad = tl.sum(h_grad.reshape(BLOCKSIZE_D, BLOCKSIZE_N) * (delta[:,None] * B[None, :]), axis = -1)
    tl.store(dx_ptr + offs_d, mask = mask_d, value = x_grad.to(tl.bfloat16))

    # B_grad -> this is correct!
    B_grad = tl.sum(h_grad.reshape(BLOCKSIZE_D, BLOCKSIZE_N) * (delta[:,None] * x[:,None]), axis = -2)
    tl.store(dB_ptr + offs_n, mask = mask_n, value=B_grad.to(tl.bfloat16))

    A_grad += intermediate * delta[:,None]
    
    for _ in range(2, num_context):

        # reduce pointer offsets
        o_ptr -= o_sq_stride
        d_out_ptr -= o_sq_stride
        delta_ptr -= delta_sq_stride
        ddelta_ptr -= delta_sq_stride
        B_ptr -= B_sq_stride
        x_ptr -= delta_sq_stride
        dx_ptr -= delta_sq_stride
        dB_ptr -= B_sq_stride

        B = tl.load(B_ptr + offs_n, mask = mask_n).to(tl.float32) # [N]
        x = tl.load(x_ptr + offs_d, mask = mask_d).to(tl.float32) # [D]

    
        h_grad = tl.exp(delta[:,None] * A).reshape(BLOCKSIZE_ND) * h_grad
        h_grad += tl.load(d_out_ptr + offs_nd, mask=mask_nd).to(tl.float32) # [ND]
        
        delta = tl.load(delta_ptr + offs_d, mask = mask_d).to(tl.float32) # [D]

        #NOTE: New
        delta_bwd_act = softplus_bwd(delta)
        delta = softplus_fwd(delta)

        h_rec = tl.load(o_ptr + offs_nd, mask=mask_nd).to(tl.float32)
        intermediate = (h_grad * h_rec).reshape(BLOCKSIZE_D, BLOCKSIZE_N) * tl.exp(delta[:,None] * A)

        delta_grad = tl.sum(intermediate * A, axis = -1) # [D,]
        delta_grad += tl.sum(h_grad.reshape(BLOCKSIZE_D, BLOCKSIZE_N) * (B[None,:] * x[:,None]), axis = -1)

        #NOTE: New
        delta_grad = delta_grad*delta_bwd_act
        tl.store(ddelta_ptr + offs_d, mask = mask_d, value=delta_grad.to(tl.bfloat16))

        x_grad = tl.sum(h_grad.reshape(BLOCKSIZE_D, BLOCKSIZE_N) * (delta[:,None] * B[None, :]), axis = -1)
        tl.store(dx_ptr + offs_d, mask = mask_d, value = x_grad.to(tl.bfloat16))

        # B_grad -> this is correct!
        B_grad = tl.sum(h_grad.reshape(BLOCKSIZE_D, BLOCKSIZE_N) * (delta[:,None] * x[:,None]), axis = -2)
        tl.store(dB_ptr + offs_n, mask = mask_n, value=B_grad.to(tl.bfloat16))

        A_grad += intermediate * delta[:,None]


    o_ptr -= o_sq_stride
    d_out_ptr -= o_sq_stride
    delta_ptr -= delta_sq_stride
    ddelta_ptr -= delta_sq_stride
    B_ptr -= B_sq_stride
    x_ptr -= delta_sq_stride
    dx_ptr -= delta_sq_stride
    dB_ptr -= B_sq_stride

    B = tl.load(B_ptr + offs_n, mask = mask_n).to(tl.float32) # [N]
    x = tl.load(x_ptr + offs_d, mask = mask_d).to(tl.float32) # [D]

    h_grad = tl.exp(delta[:,None] * A).reshape(BLOCKSIZE_ND) * h_grad
    h_grad += tl.load(d_out_ptr + offs_nd, mask=mask_nd).to(tl.float32) # [ND]

    delta_grad = tl.sum(h_grad.reshape(BLOCKSIZE_D, BLOCKSIZE_N) * (B[None,:] * x[:,None]), axis = -1)

    delta = tl.load(delta_ptr + offs_d, mask = mask_d).to(tl.float32) # [D]

    delta_bwd_act = softplus_bwd(delta)

    delta = softplus_fwd(delta)

    delta_grad = delta_grad*delta_bwd_act
    
    tl.store(ddelta_ptr + offs_d, mask = mask_d, value=delta_grad.to(tl.bfloat16))

    x_grad = tl.sum(h_grad.reshape(BLOCKSIZE_D, BLOCKSIZE_N) * (delta[:,None] * B[None, :]), axis = -1)
    tl.store(dx_ptr + offs_d, mask = mask_d, value = x_grad.to(tl.bfloat16))

    # B_grad -> this is correct!
    B_grad = tl.sum(h_grad.reshape(BLOCKSIZE_D, BLOCKSIZE_N) * (delta[:,None] * x[:,None]), axis = -2)
    tl.store(dB_ptr + offs_n, mask = mask_n, value=B_grad.to(tl.bfloat16))

    A_grad_block_ptr = tl.make_block_ptr(dA_ptr + (bs_pid * dA_bs_stride), 
                                shape = (BLOCKSIZE_D, BLOCKSIZE_N), 
                                block_shape= (BLOCKSIZE_D, BLOCKSIZE_N), 
                                offsets=(0,0), 
                                strides = (dA_row_stride, 1), order = (1,0))
                                
    tl.store(A_grad_block_ptr, A_grad)

def sequential_scan_forward(
    A: torch.Tensor,  # [d_in, n]
    delta: torch.Tensor, # [b,l,d]
    B: torch.Tensor,  # [b,l,n]
    x: torch.Tensor # [b,l,d]
) -> torch.Tensor: # [b,l, d*n]

    """Computes forward pass of a linear scan."""
    n = A.shape[1]
    b,l,d = x.shape

    o = torch.empty((b,l, d*n), dtype = x.dtype, device=x.device)

    recurrent_size = n*d

    BLOCKSIZE_LD_N = triton.next_power_of_2(n)
    BLOCKSIZE_LD_D = triton.next_power_of_2(d)
    BLOCKSIZE_ST = triton.next_power_of_2(recurrent_size)

    grid = (b,)

    #TODO: These are untuned
    match (recurrent_size):
        case _ if (recurrent_size) <= 256:
            warps = 1

        case _ if (recurrent_size) <= 512:
            warps = 2

        case _ if (recurrent_size) <= 1024:
            warps = 4

        case _ :
            warps = 8
    
    
    #fmt: off
    sequential_scan_fwd_kernel[grid](
        A,B,delta,x, o,
        A.stride(0), A.stride(1), 
        B.stride(0), B.stride(1),
        x.stride(0), x.stride(1),
        o.stride(0), o.stride(1),
        l,d,n,recurrent_size,
        BLOCKSIZE_LD_N = BLOCKSIZE_LD_N,
        BLOCKSIZE_LD_D = BLOCKSIZE_LD_D,
        BLOCKSIZE_ST = BLOCKSIZE_ST,
        num_warps = warps,
        num_stages = 1
    )
    #fmt: on
    return o

#TODO: Chunked along channel dimension is actually faster 
def sequential_scan_backward(
    A: torch.Tensor,  # [d_in, n]
    delta: torch.Tensor, # [b,l,d]
    B: torch.Tensor,  # [b,l,n]
    x: torch.Tensor, # [b,l,d]
    o: torch.Tensor, # [b,l, d*n]
    grad_out: torch.Tensor # [b,l, d*n]
) -> torch.Tensor: # [b,l, d*n]

    """Computes forward pass of a linear scan."""

    n = A.shape[1]
    b,l,d = x.shape

    A_batch_grad = torch.empty((b, d, n), dtype = torch.float32, device=A.device)
    delta_grad = torch.empty_like(delta)
    B_grad = torch.empty_like(B)
    x_grad = torch.empty_like(x)

    recurrent_size = n*d

    # we ld/st both N and D blocks
    BLOCKSIZE_N = triton.next_power_of_2(n)
    BLOCKSIZE_D = triton.next_power_of_2(d)
    BLOCKSIZE_ND = triton.next_power_of_2(recurrent_size)

    grid = (b,)

    match (recurrent_size):
        case _ if (recurrent_size) <= 256:
            warps = 1

        case _ if (recurrent_size) <= 512:
            warps = 2

        case _ if (recurrent_size) <= 1024:
            warps = 4

        case _ :
            warps = 8
    
    
    #fmt: off
    sequential_scan_bwd_kernel[grid](
        A,B, delta, x, o, grad_out,
        A_batch_grad,B_grad,delta_grad, x_grad, 
        A.stride(0),A.stride(1),
        A_batch_grad.stride(0), A_batch_grad.stride(1),
        delta.stride(0), delta.stride(1),
        B.stride(0), B.stride(1),
        o.stride(0), o.stride(1),
        l,
        d,
        n,
        recurrent_size,
        BLOCKSIZE_N,
        BLOCKSIZE_D,
        BLOCKSIZE_ND,
        num_warps = warps,
        num_stages = 1
    )
    #fmt: on
    return A_batch_grad, delta_grad, B_grad, x_grad


class DiagonalRecurrence(Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type = 'cuda', cast_inputs=torch.bfloat16) # type: ignore
    def forward(ctx,  A, delta, B, x) -> torch.Tensor:

        if not B.is_contiguous():
            B = B.contiguous()

        if not x.is_contiguous():
            x = x.contiguous()

        o = sequential_scan_forward(A,delta,B, x)

        ctx.save_for_backward(A, delta, B, x, o)

        return o

    @staticmethod
    @torch.amp.custom_bwd(device_type = 'cuda') # type: ignore
    def backward(ctx, grad_output) -> Tuple[torch.Tensor, torch.Tensor,  torch.Tensor, torch.Tensor]: # type: ignore
        (A, delta, B, x, o) = ctx.saved_tensors

        A_batch_grad, delta_grad, B_grad, x_grad = sequential_scan_backward(A, delta, B, x, o, grad_output)

        return A_batch_grad.sum(dim=0).to(A.dtype), delta_grad, B_grad, x_grad


linear_scan = DiagonalRecurrence.apply
