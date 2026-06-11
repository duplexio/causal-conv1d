import torch
from torch import Tensor

import triton
import triton.language as tl


@triton.jit
def _causal_conv1d_varlen_states(
    X,
    CU_SEQLENS,
    STATES,
    state_len,
    dim,
    stride_x_seqlen,
    stride_x_dim,
    stride_states_batch,
    stride_states_seqlen,
    stride_states_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_idx = tl.program_id(2)
    STATES += batch_idx * stride_states_batch
    end_idx = tl.load(CU_SEQLENS + batch_idx + 1)
    start_idx = tl.maximum(tl.load(CU_SEQLENS + batch_idx), end_idx - state_len)
    rows = end_idx - (tl.program_id(1) + 1) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    x = tl.load(
        X + rows[:, None] * stride_x_seqlen + cols[None, :] * stride_x_dim,
        mask=(rows[:, None] >= start_idx) & (cols[None, :] < dim),
        other=0,
    )
    rows_states = state_len - (tl.program_id(1) + 1) * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(
        STATES
        + rows_states[:, None] * stride_states_seqlen
        + cols[None, :] * stride_states_dim,
        x,
        mask=(rows_states[:, None] >= 0) & (cols[None, :] < dim),
    )


def causal_conv1d_varlen_states(
    x: Tensor, cu_seqlens: Tensor, state_len: int
) -> Tensor:
    """
    Forward pass only, does not support backward pass.
    Parameters:
        x: (total_tokens, dim)
        cu_seqlens: (batch + 1), must already be sorted. The cumulative sum of the sequence lengths, starting from 0.
        state_len: int. For each cu_seqlens, how many elements from x should be copied to the state.
            If some of those elements belong to a different sequence, the value of the states will be zero.
    Return:
        states: (batch, dim, state_len)
    """
    _, dim = x.shape
    batch = cu_seqlens.shape[0] - 1
    cu_seqlens = cu_seqlens.contiguous()
    states = torch.empty(
        batch, state_len, dim, dtype=x.dtype, device=x.device
    ).transpose(1, 2)
    BLOCK_M = min(triton.next_power_of_2(state_len), 16)
    BLOCK_N = min(triton.next_power_of_2(dim), 256)
    grid = (triton.cdiv(dim, BLOCK_N), triton.cdiv(state_len, BLOCK_M), batch)
    with torch.cuda.device(x.device.index):
        _causal_conv1d_varlen_states[grid](
            x,
            cu_seqlens,
            states,
            state_len,
            dim,
            x.stride(0),
            x.stride(1),
            states.stride(0),
            states.stride(2),
            states.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
    return states


def causal_conv1d_varlen_states_ref(
    x: Tensor, cu_seqlens: Tensor, state_len: int
) -> Tensor:
    """
    Forward pass only, does not support backward pass.
    Parameters:
        x: (total_tokens, dim)
        cu_seqlens: (batch + 1), must already be sorted. The cumulative sum of the sequence lengths, starting from 0.
        state_len: int. For each cu_seqlens, how many elements from x should be copied to the state.
            If some of those elements belong to a different sequence, the value of the states will be zero.
    Return:
        states: (batch, dim, state_len)
    """
    _, dim = x.shape
    batch = cu_seqlens.shape[0] - 1
    cu_seqlens = cu_seqlens.contiguous()
    states = torch.zeros(
        batch, state_len, dim, dtype=x.dtype, device=x.device
    ).transpose(1, 2)
    for i in range(batch):
        end_idx = cu_seqlens[i + 1]
        start_idx = torch.maximum(cu_seqlens[i], end_idx - state_len)
        states[i, :, -(end_idx - start_idx) :] = x[start_idx:end_idx].T
    return states


# ---------------------------------------------------------------------------
# Packed varlen causal conv1d -- full forward + backward via Triton
# ---------------------------------------------------------------------------
#
# Input:  x          (total_tokens, dim)   -- packed sequences
#         weight     (dim, width)
#         bias       (dim,) or None
#         cu_seqlens (batch + 1,)          -- cumulative sequence lengths
#
# Output: out        (total_tokens, dim)
#
# Semantics: For each sequence i (from cu_seqlens[i] to cu_seqlens[i+1]),
# perform a causal depthwise conv1d independently:
#   out[t, d] = bias[d] + sum_{w=0}^{width-1} weight[d, w] * x[t - (width-1) + w, d]
# where x[t, d] = 0 for t < seq_start.
#
# The kernels below use 2D tiling (BLOCK_T tokens x BLOCK_D channels) to
# maximise data reuse across the time dimension.  Each program instance
# processes a tile of BLOCK_T output tokens for BLOCK_D channels.


# Map torch dtypes to triton constexpr dtype tokens
_DTYPE_TO_TRITON = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


# ---------------------------------------------------------------------------
# In-kernel sequence boundary lookup
# ---------------------------------------------------------------------------
#
# Instead of precomputing per-token seq_start/seq_end arrays on the host,
# we resolve boundaries inside the kernel by scanning cu_seqlens directly.
# cu_seqlens has only (batch + 1) entries (typically 5-10), so the scan is
# trivially cheap.  This avoids allocating two (total_tokens,) int32 tensors
# and the host-side torch op to fill them.
#
# Triton does not support break/continue, so we iterate up to a compile-time
# MAX_SEQS bound and use tl.where to conditionally update.  Iterations
# beyond n_seqs harmlessly re-read the last cu_seqlens entry.


@triton.jit
def _seq_start_for_tokens(CU_SEQLENS, t, n_seqs, MAX_SEQS: tl.constexpr):
    """
    For a [N] vector of token positions t, return the [N] vector of
    sequence-start positions by scanning cu_seqlens[1 .. n_seqs].
    """
    seq_start = tl.zeros_like(t)
    for s in tl.static_range(MAX_SEQS):
        # s >= n_seqs: boundary == cu_seqlens[n_seqs] == total_tokens,
        # so the tl.where condition (t >= boundary) is never true for valid tokens.
        boundary = tl.load(CU_SEQLENS + tl.minimum(s + 1, n_seqs))
        seq_start = tl.where(t >= boundary, boundary, seq_start)
    return seq_start


@triton.jit
def _seq_end_for_tokens(CU_SEQLENS, t, n_seqs, MAX_SEQS: tl.constexpr):
    """
    For a [N] vector of token positions t, return the [N] vector of
    sequence-end positions by scanning cu_seqlens[n_seqs .. 1] in reverse.
    """
    total_tokens = tl.load(CU_SEQLENS + n_seqs)
    seq_end = tl.full(t.shape, value=total_tokens, dtype=tl.int32)
    for s in tl.static_range(MAX_SEQS):
        # Scan from the last boundary backwards.
        idx = n_seqs - s
        # idx <= 0: loads cu_seqlens[max(1,idx)] -- harmless, won't update.
        boundary = tl.load(CU_SEQLENS + tl.maximum(idx, 1))
        seq_end = tl.where((idx >= 1) & (t < boundary), boundary, seq_end)
    return seq_end


# ---------------------------------------------------------------------------
# Tiled forward kernel
# ---------------------------------------------------------------------------


def _fwd_configs():
    # Memory-bound: only the token-tile size matters measurably.
    return [
        triton.Config({"BLOCK_T": block_t, "BLOCK_D": 128}, num_warps=4)
        for block_t in [32, 64, 128]
    ]


@triton.autotune(configs=_fwd_configs(), key=["total_tokens_hint", "dim"])
@triton.jit
def _causal_conv1d_varlen_fwd_tiled_kernel(
    X,  # (total_tokens, dim)
    WEIGHT,  # (dim, width)
    BIAS,  # (dim,) or placeholder
    CU_SEQLENS,  # (n_seqs + 1,) int32
    OUT,  # (total_tokens, dim)
    total_tokens,
    total_tokens_hint,  # next_power_of_2(total_tokens); autotune key only
    dim,
    n_seqs,
    stride_x_tok,
    stride_x_dim,
    stride_w_dim,
    stride_w_width,
    stride_out_tok,
    stride_out_dim,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    WIDTH: tl.constexpr,
    MAX_SEQS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
):
    """
    Forward kernel with 2D tiling.
    Grid = (cdiv(total_tokens, BLOCK_T), cdiv(dim, BLOCK_D))

    Each program processes BLOCK_T output tokens x BLOCK_D channels.
    The convolution window (WIDTH taps) is iterated via static_range,
    and all BLOCK_T tokens are computed in parallel for each tap.
    Weight is loaded once per tap (shared across all BLOCK_T tokens).
    """
    pid_chunk = tl.program_id(0)
    pid_d = tl.program_id(1)

    chunk_start = pid_chunk * BLOCK_T
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # [BLOCK_D]
    d_mask = d_offs < dim

    # Token offsets for the output positions in this tile
    t_out = chunk_start + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    t_out_mask = t_out < total_tokens

    # Resolve sequence boundaries directly from cu_seqlens
    seq_starts = _seq_start_for_tokens(CU_SEQLENS, t_out, n_seqs, MAX_SEQS)

    # Load bias once (broadcast across tokens)
    if HAS_BIAS:
        bias_vals = tl.load(BIAS + d_offs, mask=d_mask, other=0.0).to(tl.float32)

    # Accumulate convolution in float32: [BLOCK_T, BLOCK_D]
    acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
    if HAS_BIAS:
        acc += bias_vals[None, :]

    for w in tl.static_range(WIDTH):
        t_in = t_out - (WIDTH - 1) + w  # [BLOCK_T]
        in_bounds = (
            (t_in[:, None] >= seq_starts[:, None])
            & t_out_mask[:, None]
            & d_mask[None, :]
        )

        x_vals = tl.load(
            X + t_in[:, None] * stride_x_tok + d_offs[None, :] * stride_x_dim,
            mask=in_bounds,
            other=0.0,
        ).to(tl.float32)  # [BLOCK_T, BLOCK_D]

        w_vals = tl.load(
            WEIGHT + d_offs * stride_w_dim + w * stride_w_width,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)  # [BLOCK_D]

        acc += x_vals * w_vals[None, :]

    if SILU_ACTIVATION:
        acc = acc * tl.sigmoid(acc)

    # Store output tile
    out_ptrs = OUT + t_out[:, None] * stride_out_tok + d_offs[None, :] * stride_out_dim
    out_mask = t_out_mask[:, None] & d_mask[None, :]
    tl.store(out_ptrs, acc.to(INPUT_DTYPE), mask=out_mask)


# ---------------------------------------------------------------------------
# Tiled fused backward kernel (dx + dweight + dbias in one pass)
# ---------------------------------------------------------------------------


def _bwd_configs():
    # Memory-bound: only the token-tile size matters measurably.
    return [
        triton.Config({"BLOCK_T": block_t, "BLOCK_D": 128}, num_warps=4)
        for block_t in [32, 64, 128]
    ]


@triton.autotune(
    configs=_bwd_configs(),
    key=["total_tokens_hint", "dim"],
    reset_to_zero=["DWEIGHT", "DBIAS"],
)
@triton.jit
def _causal_conv1d_varlen_bwd_tiled_kernel(
    DOUT,  # (total_tokens, dim)
    X,  # (total_tokens, dim)
    WEIGHT,  # (dim, width)
    BIAS,  # (dim,) or placeholder
    CU_SEQLENS,  # (n_seqs + 1,) int32
    DX,  # (total_tokens, dim) -- output
    DWEIGHT,  # (dim, width) float32 -- output, atomic adds
    DBIAS,  # (dim,) float32 or placeholder -- output, atomic adds
    total_tokens,
    total_tokens_hint,  # next_power_of_2(total_tokens); autotune key only
    dim,
    n_seqs,
    stride_x_tok,
    stride_x_dim,
    stride_dout_tok,
    stride_dout_dim,
    stride_w_dim,
    stride_w_width,
    stride_dx_tok,
    stride_dx_dim,
    stride_dw_dim,
    stride_dw_width,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    WIDTH: tl.constexpr,
    MAX_SEQS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
):
    """
    Fused backward kernel: computes dx, dweight, and dbias in a single pass.
    Grid = (cdiv(total_tokens, BLOCK_T), cdiv(dim, BLOCK_D))

    Each program handles a [BLOCK_T, BLOCK_D] tile of tokens x channels.

    Data layout:
      - For dweight/dbias: we need dout_eff[t] and x[t-W+1..t] for t in tile.
        We load the same x data as the forward kernel.
      - For dx: dx[t] = sum_j weight[W-1-j] * dout_eff[t+j], so we need
        dout_eff for positions [chunk_start .. chunk_start + BLOCK_T + W - 2].
        We load an extended tile of dout and x for the extra W-1 positions.
    """
    pid_chunk = tl.program_id(0)
    pid_d = tl.program_id(1)

    chunk_start = pid_chunk * BLOCK_T
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # [BLOCK_D]
    d_mask = d_offs < dim

    # Token offsets for this tile's output positions
    t_out = chunk_start + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    t_out_mask = t_out < total_tokens

    # Resolve sequence boundaries directly from cu_seqlens
    seq_starts = _seq_start_for_tokens(CU_SEQLENS, t_out, n_seqs, MAX_SEQS)
    seq_ends = _seq_end_for_tokens(CU_SEQLENS, t_out, n_seqs, MAX_SEQS)

    # Load weight vectors
    w_ptrs_base = WEIGHT + d_offs * stride_w_dim

    # Load bias for SiLU recomputation
    if HAS_BIAS & SILU_ACTIVATION:
        bias_vals = tl.load(BIAS + d_offs, mask=d_mask, other=0.0).to(tl.float32)

    # =====================================================================
    # Step 1: Load dout tile and compute effective dout (with SiLU factor)
    #         for the BLOCK_T positions in this tile.
    # =====================================================================
    dout_tile = tl.load(
        DOUT + t_out[:, None] * stride_dout_tok + d_offs[None, :] * stride_dout_dim,
        mask=t_out_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)  # [BLOCK_T, BLOCK_D]

    if SILU_ACTIVATION:
        # Recompute pre-activation from x and weight (all in registers)
        pre_act = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
        if HAS_BIAS:
            pre_act += bias_vals[None, :]
        for w in tl.static_range(WIDTH):
            t_in = t_out - (WIDTH - 1) + w  # [BLOCK_T]
            in_bounds = (
                (t_in[:, None] >= seq_starts[:, None])
                & t_out_mask[:, None]
                & d_mask[None, :]
            )
            x_for_pre = tl.load(
                X + t_in[:, None] * stride_x_tok + d_offs[None, :] * stride_x_dim,
                mask=in_bounds,
                other=0.0,
            ).to(tl.float32)
            w_val = tl.load(
                w_ptrs_base + w * stride_w_width, mask=d_mask, other=0.0
            ).to(tl.float32)
            pre_act += x_for_pre * w_val[None, :]
        sig = tl.sigmoid(pre_act)
        silu_grad = sig + pre_act * sig * (1.0 - sig)
        dout_tile = dout_tile * silu_grad

    # =====================================================================
    # Step 2: Accumulate dweight across BLOCK_T tokens -> one atomic per tile
    # =====================================================================
    for w in tl.static_range(WIDTH):
        t_in = t_out - (WIDTH - 1) + w  # [BLOCK_T]
        in_bounds = (
            (t_in[:, None] >= seq_starts[:, None])
            & t_out_mask[:, None]
            & d_mask[None, :]
        )
        x_vals = tl.load(
            X + t_in[:, None] * stride_x_tok + d_offs[None, :] * stride_x_dim,
            mask=in_bounds,
            other=0.0,
        ).to(tl.float32)  # [BLOCK_T, BLOCK_D]
        dw_contrib = tl.sum(dout_tile * x_vals, axis=0)  # [BLOCK_D]
        tl.atomic_add(
            DWEIGHT + d_offs * stride_dw_dim + w * stride_dw_width,
            dw_contrib,
            mask=d_mask,
        )

    # =====================================================================
    # Step 3: Accumulate dbias across BLOCK_T tokens
    # =====================================================================
    if HAS_BIAS:
        db_contrib = tl.sum(dout_tile, axis=0)  # [BLOCK_D]
        tl.atomic_add(DBIAS + d_offs, db_contrib, mask=d_mask)

    # =====================================================================
    # Step 4: Compute dx via transposed convolution
    #
    # dx[t] = sum_{j=0}^{W-1} weight[W-1-j] * dout_eff[t+j]
    # where dout_eff[t+j] must be in the same sequence as t.
    #
    # We need dout_eff for positions [chunk_start .. chunk_start+BLOCK_T+W-2].
    # dout_tile covers [chunk_start .. chunk_start+BLOCK_T-1].
    # We need W-1 extra positions beyond the tile.
    # =====================================================================

    # For each shift j, load dout_eff[t_out + j] from global memory and
    # recompute the SiLU factor.  Each shifted load is a coalesced [BLOCK_T, BLOCK_D]
    # read, and the SiLU recomputation amortises its x loads across BLOCK_T tokens.
    dx_tile = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)

    for j in tl.static_range(WIDTH):
        w_val = tl.load(
            w_ptrs_base + (WIDTH - 1 - j) * stride_w_width,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)  # [BLOCK_D]

        t_shifted = t_out + j  # [BLOCK_T]
        t_shifted_valid = (t_shifted < total_tokens) & t_out_mask

        # Load dout at shifted positions
        dout_shifted = tl.load(
            DOUT
            + t_shifted[:, None] * stride_dout_tok
            + d_offs[None, :] * stride_dout_dim,
            mask=t_shifted_valid[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)  # [BLOCK_T, BLOCK_D]

        if SILU_ACTIVATION:
            # Find seq_start for shifted positions
            ss_shifted = _seq_start_for_tokens(
                CU_SEQLENS, t_shifted, n_seqs, MAX_SEQS
            )  # [BLOCK_T]

            pre_act_s = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
            if HAS_BIAS:
                pre_act_s += bias_vals[None, :]
            for w2 in tl.static_range(WIDTH):
                t_in_s = t_shifted - (WIDTH - 1) + w2
                in_bounds_s = (
                    (t_in_s[:, None] >= ss_shifted[:, None])
                    & t_shifted_valid[:, None]
                    & d_mask[None, :]
                )
                x_s = tl.load(
                    X + t_in_s[:, None] * stride_x_tok + d_offs[None, :] * stride_x_dim,
                    mask=in_bounds_s,
                    other=0.0,
                ).to(tl.float32)
                w_val2 = tl.load(
                    w_ptrs_base + w2 * stride_w_width, mask=d_mask, other=0.0
                ).to(tl.float32)
                pre_act_s += x_s * w_val2[None, :]
            sig_s = tl.sigmoid(pre_act_s)
            silu_grad_s = sig_s + pre_act_s * sig_s * (1.0 - sig_s)
            dout_shifted = dout_shifted * silu_grad_s

        # Mask: t_out + j must be within the same sequence as t_out
        dout_ok = (
            (t_shifted[:, None] < seq_ends[:, None])
            & t_shifted_valid[:, None]
            & d_mask[None, :]
        )
        dout_shifted = tl.where(dout_ok, dout_shifted, 0.0)

        dx_tile += dout_shifted * w_val[None, :]

    # Store dx
    dx_ptrs = DX + t_out[:, None] * stride_dx_tok + d_offs[None, :] * stride_dx_dim
    dx_mask = t_out_mask[:, None] & d_mask[None, :]
    tl.store(dx_ptrs, dx_tile.to(INPUT_DTYPE), mask=dx_mask)


# ---------------------------------------------------------------------------
# Autograd wrapper
# ---------------------------------------------------------------------------


class CausalConv1dVarlenFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, cu_seqlens, activation):
        """
        x: (total_tokens, dim)
        weight: (dim, width)
        bias: (dim,) or None
        cu_seqlens: (batch + 1,) int32
        activation: None or "silu" or "swish"
        """
        if activation not in [None, "silu", "swish"]:
            raise NotImplementedError("activation must be None, silu, or swish")

        total_tokens, dim = x.shape
        dim_w, width = weight.shape
        assert dim == dim_w, f"x dim {dim} != weight dim {dim_w}"
        assert width in (2, 3, 4), f"width must be 2, 3, or 4, got {width}"

        x = x.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        cu_seqlens = cu_seqlens.contiguous()

        silu = activation in ("silu", "swish")

        n_seqs = cu_seqlens.shape[0] - 1
        MAX_SEQS = triton.next_power_of_2(n_seqs)

        out = torch.empty_like(x)

        input_triton_dtype = _DTYPE_TO_TRITON[x.dtype]

        def fwd_grid(META):
            return (
                triton.cdiv(total_tokens, META["BLOCK_T"]),
                triton.cdiv(dim, META["BLOCK_D"]),
            )

        with torch.cuda.device(x.device.index):
            _causal_conv1d_varlen_fwd_tiled_kernel[fwd_grid](
                x,
                weight,
                bias if bias is not None else x,  # placeholder, won't be read
                cu_seqlens,
                out,
                total_tokens,
                triton.next_power_of_2(total_tokens),
                dim,
                n_seqs,
                x.stride(0),
                x.stride(1),
                weight.stride(0),
                weight.stride(1),
                out.stride(0),
                out.stride(1),
                HAS_BIAS=bias is not None,
                SILU_ACTIVATION=silu,
                WIDTH=width,
                MAX_SEQS=MAX_SEQS,
                INPUT_DTYPE=input_triton_dtype,
            )

        ctx.save_for_backward(x, weight, bias, cu_seqlens)
        ctx.silu = silu
        ctx.total_tokens = total_tokens
        ctx.dim = dim
        ctx.width = width
        return out

    @staticmethod
    def backward(ctx, dout):
        x, weight, bias, cu_seqlens = ctx.saved_tensors
        silu = ctx.silu
        total_tokens = ctx.total_tokens
        dim = ctx.dim
        width = ctx.width

        n_seqs = cu_seqlens.shape[0] - 1
        MAX_SEQS = triton.next_power_of_2(n_seqs)

        dout = dout.contiguous()

        input_triton_dtype = _DTYPE_TO_TRITON[x.dtype]

        dx = torch.empty_like(x)
        dweight = torch.zeros(dim, width, dtype=torch.float32, device=x.device)
        dbias = (
            torch.zeros(dim, dtype=torch.float32, device=x.device)
            if bias is not None
            else None
        )

        def bwd_grid(META):
            return (
                triton.cdiv(total_tokens, META["BLOCK_T"]),
                triton.cdiv(dim, META["BLOCK_D"]),
            )

        with torch.cuda.device(x.device.index):
            _causal_conv1d_varlen_bwd_tiled_kernel[bwd_grid](
                dout,
                x,
                weight,
                bias if bias is not None else x,  # placeholder
                cu_seqlens,
                dx,
                dweight,
                dbias if dbias is not None else dweight,  # placeholder
                total_tokens,
                triton.next_power_of_2(total_tokens),
                dim,
                n_seqs,
                x.stride(0),
                x.stride(1),
                dout.stride(0),
                dout.stride(1),
                weight.stride(0),
                weight.stride(1),
                dx.stride(0),
                dx.stride(1),
                dweight.stride(0),
                dweight.stride(1),
                HAS_BIAS=bias is not None,
                SILU_ACTIVATION=silu,
                WIDTH=width,
                MAX_SEQS=MAX_SEQS,
                INPUT_DTYPE=input_triton_dtype,
            )

        dweight = dweight.to(weight.dtype)
        if dbias is not None:
            dbias = dbias.to(bias.dtype)

        return dx, dweight, dbias, None, None


def causal_conv1d_varlen_fn(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    activation: str | None = None,
) -> Tensor:
    """
    Causal depthwise conv1d on packed variable-length sequences.

    Parameters:
        x: (total_tokens, dim)
        weight: (dim, width)  -- width must be 2, 3, or 4
        bias: (dim,) or None
        cu_seqlens: (batch + 1,), int32, sorted cumulative sequence lengths starting from 0.
        activation: None, "silu", or "swish"

    Returns:
        out: (total_tokens, dim)
    """
    return CausalConv1dVarlenFn.apply(x, weight, bias, cu_seqlens, activation)


def causal_conv1d_varlen_ref(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    activation: str | None = None,
) -> Tensor:
    """
    Reference implementation: split into individual segments and run causal_conv1d_ref on each.

    Parameters:
        x: (total_tokens, dim)
        weight: (dim, width)
        bias: (dim,) or None
        cu_seqlens: (batch + 1,)
        activation: None, "silu", or "swish"

    Returns:
        out: (total_tokens, dim)
    """
    from causal_conv1d.causal_conv1d_interface import causal_conv1d_ref

    assert cu_seqlens is not None, "cu_seqlens is required"
    total_tokens, dim = x.shape
    batch = cu_seqlens.shape[0] - 1
    out = torch.empty_like(x)

    for i in range(batch):
        s = cu_seqlens[i].item()
        e = cu_seqlens[i + 1].item()
        # causal_conv1d_ref expects (batch, dim, seqlen)
        x_seg = x[s:e].T.unsqueeze(0)  # (1, dim, seg_len)
        out_seg = causal_conv1d_ref(x_seg, weight, bias, activation=activation)
        out[s:e] = out_seg.squeeze(0).T  # back to (seg_len, dim)

    return out
