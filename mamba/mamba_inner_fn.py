import torch
import torch.nn.functional as F
from torch.autograd import Function

from mamba.causal_conv1d import causal_conv1d_fn_bwd, causal_conv1d_fn_fwd
from mamba.scan_fused import parallel_scan_bwd, parallel_scan_fwd


def silu_bwd(x):
    return x.sigmoid() + (x * x.sigmoid() * (1 - x.sigmoid()))


class MambaInnerFn(Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.bfloat16)  # type: ignore
    def forward(
        ctx,
        x_and_res,
        conv1d_w,
        conv1d_b,
        A_log,
        D,
        x_proj_w,
        dt_proj_w,
        dt_proj_b,
        out_proj_w,
    ):
        ctx.save_for_backward(
            x_and_res,
            conv1d_w,
            conv1d_b,
            A_log,
            D,
            x_proj_w,
            dt_proj_w,
            dt_proj_b,
            out_proj_w,
        )

        # x_and_res = F.linear(x, in_proj_w)

        (x, res) = torch.chunk(x_and_res, chunks=2, dim=-1)

        x = causal_conv1d_fn_fwd(x.mT.contiguous(), conv1d_w, conv1d_b, act=1)

        A = -torch.exp(A_log.float())
        D = D.float()

        x_dbl = torch.einsum("bdl,md->bml", x, x_proj_w)

        n = A.shape[1]
        dt_rank = dt_proj_w.shape[1]
        (delta, B, C) = x_dbl.split(split_size=[dt_rank, n, n], dim=1)

        delta = torch.einsum("brl,nr->bnl", delta, dt_proj_w) + dt_proj_b[None, :, None]

        scan_out = parallel_scan_fwd(
            A,
            delta.contiguous(),
            B.contiguous(),
            C.contiguous(),
            x.contiguous(),
        )
        y = scan_out.mT.contiguous()

        y = y + x.mT.contiguous() * D

        y = y * F.silu(res)

        out = F.linear(y.to(x.dtype), out_proj_w)

        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")  # type: ignore
    def backward(ctx, grad_output):
        (
            x,
            in_proj_w,
            conv1d_w,
            conv1d_b,
            A_log,
            D,
            x_proj_w,
            dt_proj_w,
            dt_proj_b,
            out_proj_w,
        ) = ctx.saved_tensors

        # dconv1d_w = torch.empty_like(conv1d_w)
        # dconv1d_b = torch.empty_like(conv1d_b)
        # dA_log = torch.empty_like(A_log)
        # dD = torch.empty_like(D)
        # dx_proj_w = torch.empty_like(x_proj_w)
        # ddt_proj_w = torch.empty_like(dt_proj_w)
        # ddt_proj_b = torch.empty_like(dt_proj_b)

        # -------------------
        # Re-run forward pass
        # -------------------

        x_and_res = F.linear(x, in_proj_w)
        (x, res) = torch.chunk(x_and_res, chunks=2, dim=-1)

        x_conv_out = causal_conv1d_fn_fwd(x.mT.contiguous(), conv1d_w, conv1d_b, act=1)

        A = -torch.exp(A_log.float())
        D = D.float()

        x_dbl = torch.einsum("bdl,md->bml", x_conv_out, x_proj_w)

        n = A.shape[1]
        dt_rank = dt_proj_w.shape[1]
        (delta, B, C) = x_dbl.split(split_size=[dt_rank, n, n], dim=1)

        delta_act = (
            torch.einsum("brl,nr->bnl", delta, dt_proj_w) + dt_proj_b[None, :, None]
        )

        scan_out = parallel_scan_fwd(
            A,
            delta_act.contiguous(),
            B.contiguous(),
            C.contiguous(),
            x_conv_out.contiguous(),
        )
        y = scan_out.mT.contiguous()

        y = y + x_conv_out.mT * D

        y_f = y * F.silu(res)

        # -------------
        # Backward Pass
        # -------------

        dy = grad_output @ out_proj_w

        dres = dy * y * silu_bwd(res)

        dout_proj_w = torch.sum(grad_output.mT @ y_f.to(x_conv_out.dtype), dim=0)

        dy = dy * F.silu(res)

        dD = torch.sum(dy * x_conv_out.mT, dim=(0, 1))

        dx_conv_out = dy * D

        A_grad, B_grad, C_grad, delta_grad, x_grad = parallel_scan_bwd(
            A,
            delta_act.contiguous(),
            B.contiguous(),
            C.contiguous(),
            x_conv_out.contiguous(),
            dy,
        )

        dA_log = A_grad * A

        ddt_proj_w = torch.einsum("bdl,brl->dr", delta_grad, delta)
        ddt_proj_b = torch.einsum("bdl->d", delta_grad)
        ddelta = torch.einsum("bdl,dr->brl", delta_grad, dt_proj_w)

        xdbl_act_grad = torch.cat([ddelta, B_grad, C_grad], dim=1)
        dx_proj_w = torch.einsum("bml,bdl->md", xdbl_act_grad, x_conv_out)
        dx_proj_act = torch.einsum("bml,md->bdl", xdbl_act_grad, x_proj_w)

        x_grad = dx_conv_out.mT + x_grad + dx_proj_act

        dx_pre_conv, dconv1d_w, dconv1d_b = causal_conv1d_fn_bwd(
            x.mT.contiguous(), conv1d_w, conv1d_b, x_grad.contiguous(), act=1
        )

        dx = torch.cat([dx_pre_conv.mT.contiguous(), dres], dim=-1)

        return (
            dx,
            dconv1d_w,
            dconv1d_b,
            dA_log,
            dD,
            dx_proj_w,
            ddt_proj_w,
            ddt_proj_b,
            dout_proj_w,
        )


if __name__ == "__main__":
    from mamba.mamba import MambaBlock, MambaConfig

    hidden = 128
    intermediate = 256
    state = 16
    rank = 64

    test_config = MambaConfig(8192, hidden, intermediate, state, 1, rank)

    block = MambaBlock(test_config)
    block.cuda()
    b, l, d = 4, 1024, hidden

    activation = torch.randn(
        (b, l, d), device="cuda", dtype=torch.float32, requires_grad=True
    )

    torch.nn.init.ones_(block.conv1d.weight)
    torch.nn.init.ones_(block.conv1d.bias)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        reference_out, _ = block(activation, None)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        ckpt_out = MambaInnerFn.apply(
            activation,
            block.in_proj.weight,
            block.conv1d.weight,
            block.conv1d.bias,
            block.A_log,
            block.D,
            block.x_proj.weight,
            block.dt_proj.weight,
            block.dt_proj.bias,
            block.out_proj.weight,
        )

    def relative_error(x: torch.Tensor, y: torch.Tensor) -> float:
        return (torch.linalg.norm(x - y) / torch.linalg.norm(y)).item()

    print(f"forward pass: {relative_error(ckpt_out, reference_out)}")

    dy = 0.1 * torch.randn_like(reference_out)

    reference_out.backward(dy, retain_graph=True)

    doutproj_w_torch = block.out_proj.weight.grad.clone()
    dAlog_torch = block.A_log.grad.clone()
    ddelta_w_torch = block.dt_proj.weight.grad.clone()
    ddelta_b_torch = block.dt_proj.bias.grad.clone()
    dD_torch = block.D.grad.clone()
    dx_proj_w_torch = block.x_proj.weight.grad.clone()
    dconv1d_w_torch = block.conv1d.weight.grad.clone()
    dconv1d_bias_torch = block.conv1d.bias.grad.clone()
    dact_torch = activation.grad.clone()

    activation.grad = None  # TODO

    # Done
    block.in_proj.weight.grad = None
    block.A_log.grad = None
    block.D.grad = None
    block.dt_proj.weight.grad = None
    block.dt_proj.bias.grad = None
    block.out_proj.weight.grad = None
    block.x_proj.weight.grad = None
    block.conv1d.weight.grad = None
    block.conv1d.bias.grad = None

    ckpt_out.backward(dy, retain_graph=True)

    doutproj_w_triton = block.out_proj.weight.grad.clone()
    dAlog_triton = block.A_log.grad.clone()
    ddelta_w_triton = block.dt_proj.weight.grad.clone()
    ddelta_b_triton = block.dt_proj.bias.grad.clone()
    dD_triton = block.D.grad.clone()
    dx_proj_w_triton = block.x_proj.weight.grad.clone()

    dconv1d_w_triton = block.conv1d.weight.grad.clone()
    dconv1d_bias_triton = block.conv1d.bias.grad.clone()

    dact_triton = activation.grad.clone()

    print(f"dact: {relative_error(dact_triton, dact_torch)}")

    # Done
    print(f"dconv1d_w: {relative_error(dconv1d_w_triton, dconv1d_w_torch)}")
    print(f"dconv1d_b: {relative_error(dconv1d_bias_triton, dconv1d_bias_torch)}")
    print(f"dx_proj: {relative_error(dx_proj_w_triton, dx_proj_w_torch)}")
    print(f"dD: {relative_error(dD_triton, dD_torch)}")
    print(f"ddelta_b: {relative_error(ddelta_b_triton, ddelta_b_torch)}")
    print(f"ddelta_w: {relative_error(ddelta_w_triton, ddelta_w_torch)}")
    print(f"dxproj_w: {relative_error(doutproj_w_triton, doutproj_w_torch)}")
    print(f"dAlog: {relative_error(dAlog_triton, dAlog_torch)}")