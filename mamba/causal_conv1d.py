import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd import Function


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def dsilu(x):
    sig = tl.sigmoid(x)
    return sig + (x * sig * (1 - sig))


@triton.jit
def causal_conv_forward_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    x_stride_b,
    x_stride_d,
    w_stride_d,
    seqlen: tl.constexpr,
    kernel_t: tl.constexpr,
    act: tl.constexpr,
):
    b_pid = tl.program_id(0)
    d_pid = tl.program_id(1)

    # offset to beginning of block
    x_ptr += b_pid * x_stride_b + d_pid * x_stride_d
    y_ptr += b_pid * x_stride_b + d_pid * x_stride_d
    w_ptr += d_pid * w_stride_d

    b = tl.load(b_ptr + d_pid)

    y = tl.zeros([seqlen], dtype=tl.float32)

    for t in range(0, kernel_t):
        # Succesively load x in shifts

        w_load = (kernel_t - 1) - t

        t_off = t

        w = tl.load(w_ptr + w_load)

        # NOTE: Not vectorized since these are loads from global

        x = tl.load(
            x_ptr - t_off + tl.arange(0, seqlen),
            mask=tl.arange(0, seqlen) >= t_off,
            other=0.0,
        )
        y += x * w
    y += b

    if act == 1:
        tl.store(y_ptr + tl.arange(0, seqlen), silu(y).to(tl.bfloat16))
    else:
        tl.store(y_ptr + tl.arange(0, seqlen), y.to(tl.bfloat16))


@triton.jit
def causal_conv_backward_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    grad_out_ptr,
    scratch_ptr,
    dx_ptr,
    dw_ptr,
    db_ptr,
    x_stride_b,
    x_stride_d,
    w_stride_d,
    dw_stride_b,
    dw_stride_d,
    db_stride_b,
    seqlen: tl.constexpr,
    kernel_t: tl.constexpr,
    act: tl.const,
):
    b_pid = tl.program_id(0)
    d_pid = tl.program_id(1)

    # offset to beginning of block
    x_ptr += b_pid * x_stride_b + d_pid * x_stride_d
    dx_ptr += b_pid * x_stride_b + d_pid * x_stride_d
    w_ptr += d_pid * w_stride_d

    grad_out_ptr += b_pid * x_stride_b + d_pid * x_stride_d
    scratch_ptr += b_pid * x_stride_b + d_pid * x_stride_d

    b = tl.load(b_ptr + d_pid)

    y_fwd_no_act = tl.zeros([seqlen], dtype=tl.float32)

    x_grad = tl.zeros_like(y_fwd_no_act)

    grad_out = tl.load(grad_out_ptr + tl.arange(0, seqlen))

    x = tl.load(x_ptr + tl.arange(0, seqlen), cache_modifier=".ca")

    for t in range(0, kernel_t):
        # Succesively load x in shifts
        w_load = (kernel_t - 1) - t

        w = tl.load(w_ptr + w_load)

        offs_t = tl.arange(0, seqlen) - t

        # NOTE: Not vectorized since these are loads from global

        x = tl.load(
            x_ptr + offs_t,
            mask=offs_t >= 0,
            other=0.0,
        )
        y_fwd_no_act += x * w

    y_fwd_no_act += b

    if act:
        y_grad = grad_out * dsilu(y_fwd_no_act.to(tl.float32))  # [seqlen]
    else:
        y_grad = grad_out.to(tl.float32)

    # NOTE: this is the cache-modifier we want for scratchpad, doesn't help much though
    tl.store(
        scratch_ptr + tl.arange(0, seqlen), y_grad.to(tl.bfloat16), cache_modifier=".wb"
    )

    x = tl.load(x_ptr + tl.arange(0, seqlen), cache_modifier=".ca").to(tl.float32)

    db_ptr += b_pid * db_stride_b + d_pid
    db_grad = tl.sum(y_grad)
    tl.store(db_ptr, db_grad.to(tl.bfloat16))

    dw_ptr += b_pid * dw_stride_b + d_pid * dw_stride_d

    for t in range(0, kernel_t):
        offs_t = t + tl.arange(0, seqlen)

        # NOTE: Not vectorized since these are loads from global
        y_grad_rolled = tl.load(scratch_ptr + offs_t, mask=offs_t < seqlen).to(
            tl.bfloat16
        )

        w = tl.load(w_ptr + (kernel_t - 1 - t))

        x_grad += y_grad_rolled * w
        val = tl.sum(y_grad_rolled * x)

        tl.store(dw_ptr + (kernel_t - 1 - t), val.to(tl.bfloat16))

    tl.store(dx_ptr + tl.arange(0, seqlen), value=x_grad.to(tl.bfloat16))


# docstring from https://github.com/Dao-AILab/causal-conv1d :)
def causal_conv1d_fn_fwd(x, weight, bias=None, act: int = 1):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    activation: either None or "silu"

    out: (batch, dim, seqlen)
    """

    y = torch.empty_like(x)

    b = x.shape[0]
    d = x.shape[1]

    seqlen = x.shape[-1]
    kernel_t = weight.shape[-1]

    grid = (b, d)

    if not x.is_contiguous():
        x = x.contiguous()

    match seqlen:
        case _ if (seqlen) <= 2048:
            warps = 1

        case _ if (seqlen) <= 4096:
            warps = 2

        case _ if (seqlen) <= 8192:
            warps = 4

        case _:
            warps = 8

    causal_conv_forward_kernel[grid](
        x,
        weight,
        bias,
        y,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        seqlen,
        kernel_t,
        act=act,
        enable_fp_fusion=True,
        num_warps=warps,
    )

    return y


def causal_conv1d_fn_bwd(x, weight, bias, grad_out, act):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    grad_out: (batch, dim, seqlen)
    act: int (act=1 for 'silu' activation)

    dx: (batch, dim, seqlen)
    dw: (dim, width)
    dbias: (dim,)
    """

    b = x.shape[0]
    d = x.shape[1]

    seqlen = x.shape[-1]
    kernel_t = weight.shape[-1]

    grid = (b, d)

    x_grad = torch.empty_like(x)
    bias_grad = torch.empty(
        (b, bias.shape[0]),
        dtype=bias.dtype,
        device=bias.device,
    )
    weight_grad = torch.zeros(
        size=(b, weight.shape[0], kernel_t),
        dtype=weight.dtype,
        device=weight.device,
    )

    match seqlen:
        case _ if (seqlen) <= 2048:
            warps = 1

        case _ if (seqlen) <= 4096:
            warps = 2

        case _ if (seqlen) <= 8192:
            warps = 4

        case _:
            warps = 8

    grad_scratchpad = torch.empty_like(grad_out, dtype=torch.float32)

    causal_conv_backward_kernel[grid](
        x,
        weight,
        bias,
        grad_out,
        grad_scratchpad,
        x_grad,
        weight_grad,
        bias_grad,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight_grad.stride(0),
        weight_grad.stride(1),
        bias_grad.stride(0),
        seqlen,
        kernel_t,
        act=act,
        enable_fp_fusion=True,
        num_warps=warps,
    )
    return x_grad, weight_grad.sum(dim=0).unsqueeze(1), bias_grad.sum(dim=0)


class CausalConv(Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)  # type: ignore
    def forward(ctx, x, weight, bias, act):
        ctx.save_for_backward(x, weight, bias)
        ctx.act = act

        return causal_conv1d_fn_fwd(x, weight, bias, act=act)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")  # type: ignore
    def backward(ctx, grad_output):
        (x, weight, bias) = ctx.saved_tensors

        x_grad, weight_grad, bias_grad = causal_conv1d_fn_bwd(
            x, weight, bias, grad_output, act=ctx.act
        )

        return x_grad, weight_grad, bias_grad, None


def causal_conv(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, act: int
) -> torch.Tensor:
    return CausalConv.apply(x, weight, bias, act)


if __name__ == "__main__":
    b = 4
    d = 128
    l = 4096
    width = 4

    temporal_width = 4
    conv1d = torch.nn.Conv1d(
        in_channels=d,
        out_channels=d,
        bias=True,
        kernel_size=width,
        groups=d,
        padding=width - 1,
    )
    conv1d.cuda()

    x = torch.randn((b, d, l), device="cuda", dtype=torch.float32, requires_grad=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out_ref = F.silu(conv1d(x))[..., :l]

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out_tl = causal_conv(x, conv1d.weight, conv1d.bias, act=1)
        # out_tl = causal_conv1d_fn_fwd(x, conv1d.weight, conv1d.bias, act=1)
    dy = torch.ones_like(out_ref)

    print(torch.linalg.norm(out_ref - out_tl) / torch.linalg.norm(out_ref))

    # # Backward pass - PyTorch
    out_ref.backward(dy, retain_graph=True)
    dx_torch = x.grad.clone()
    dw_torch = conv1d.weight.grad.clone()
    db_torch = conv1d.bias.grad.clone()

    # Reset grads for manual implementation
    x.grad = None
    conv1d.weight.grad = None
    conv1d.bias.grad = None

    # Backward pass - Manual implementation
    out_tl.backward(dy, retain_graph=True)

    def relative_error(x: torch.Tensor, y: torch.Tensor) -> float:
        return (torch.linalg.norm(x - y) / torch.linalg.norm(y)).item()

    dx_manual = x.grad.clone()
    dw_manual = conv1d.weight.grad.clone()
    db_manual = conv1d.bias.grad.clone()

    print(f"dx: {relative_error(dx_manual, dx_torch)}")
    print(f"dW: {relative_error(dw_manual, dw_torch)}")
    print(f"db: {relative_error(db_manual, db_torch)}")
