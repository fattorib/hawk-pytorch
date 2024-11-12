import torch
import torch.nn.functional as F
from torch.autograd import Function

from .causal_conv1d import causal_conv1d_fn_bwd, causal_conv1d_fn_fwd
from .scan_fused import parallel_scan_bwd, parallel_scan_fwd


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
        print(x_and_res.shape)
        print(
            conv1d_w.shape,
            conv1d_b.shape,
            x_proj_w.shape,
            dt_proj_w.shape,
            dt_proj_b.shape,
            out_proj_w.shape,
        )
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
    def backward(ctx, grad_output):  # type: ignore
        (
            x_and_res,
            conv1d_w,
            conv1d_b,
            A_log,
            D,
            x_proj_w,
            dt_proj_w,
            dt_proj_b,
            out_proj_w,
        ) = ctx.saved_tensors

        # -------------------
        # Re-run forward pass
        # -------------------

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

        dres = dy * y * silu_bwd(res.float())

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
            dy.mT.contiguous(),
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

        dx = torch.cat(
            [dx_pre_conv.mT.contiguous(), dres], dim=-1
        )  # TODO: This is pretty slow, probably better to fuse with bwd on cc

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
