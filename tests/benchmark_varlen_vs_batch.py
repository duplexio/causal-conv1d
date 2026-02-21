#!/usr/bin/env python
"""
Benchmark + correctness check: causal_conv1d_varlen_fn vs causal_conv1d_fn

Uses *identical* data so outputs are directly comparable.  For each (batch, dim,
seqlen, width, activation, bias) configuration the script:

  1. Creates shared random data (same seed).
  2. Runs forward through both the batched CUDA kernel (causal_conv1d_fn with
     seq_idx to mimic packed sequences) and the packed varlen Triton kernel
     (causal_conv1d_varlen_fn).
  3. Runs backward through both.
  4. Compares outputs, dx, dweight, dbias and reports max-abs / mean-abs diffs.
  5. Reports forward, backward, and fwd+bwd timing for both paths.
"""

import argparse
import statistics
import sys

import torch
from einops import rearrange

from causal_conv1d.causal_conv1d_interface import causal_conv1d_fn
from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_fn


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


def _do_bench(fn, warmup=25, rep=100, iters=10):
    """Return median kernel time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start_ev.record()
        for _ in range(rep):
            fn()
        end_ev.record()
        torch.cuda.synchronize()
        times.append(start_ev.elapsed_time(end_ev) / rep)
    return statistics.median(times)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def make_shared_data(batch, dim, seqlen, width, dtype, has_bias, device="cuda", seed=0):
    """
    Create a single random dataset and derive both batched and varlen views.

    Batched path:
        x_batch: (batch, dim, seqlen)  -- channel-last storage so seq_idx works
        weight_batch, bias_batch, seq_idx

    Varlen path:
        x_varlen: (total_tokens, dim)
        weight_varlen, bias_varlen, cu_seqlens
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    total_tokens = batch * seqlen

    # --- raw data (fp32, then cast) ---
    x_raw = torch.randn(batch, seqlen, dim, device=device, dtype=dtype)
    weight_raw = torch.randn(dim, width, device=device, dtype=torch.float32)
    bias_raw = (
        torch.randn(dim, device=device, dtype=torch.float32) if has_bias else None
    )

    # --- batched path ---
    # causal_conv1d_fn expects (batch, dim, seqlen) and we use channel-last so
    # that seq_idx is supported (identical to test_causal_conv1d_varlen).
    x_batch = rearrange(x_raw.clone(), "b s d -> b d s").requires_grad_()
    weight_batch = weight_raw.clone().requires_grad_()
    bias_batch = bias_raw.clone().requires_grad_() if has_bias else None
    # seq_idx: every element in a batch item belongs to the same sequence (index 0)
    seq_idx = torch.zeros(batch, seqlen, dtype=torch.int32, device=device)

    # --- varlen path ---
    # Pack all batch items into a single (total_tokens, dim) tensor.
    x_varlen = rearrange(x_raw.clone(), "b s d -> (b s) d").requires_grad_()
    weight_varlen = weight_raw.clone().requires_grad_()
    bias_varlen = bias_raw.clone().requires_grad_() if has_bias else None
    cu_seqlens = torch.arange(
        0, total_tokens + 1, seqlen, dtype=torch.int32, device=device
    )

    # --- gradient tensor (must match batched output shape) ---
    torch.manual_seed(seed + 1)
    g_batch = torch.randn(batch, dim, seqlen, device=device, dtype=dtype)
    g_varlen = rearrange(g_batch.clone(), "b d s -> (b s) d")

    return dict(
        x_batch=x_batch,
        weight_batch=weight_batch,
        bias_batch=bias_batch,
        seq_idx=seq_idx,
        x_varlen=x_varlen,
        weight_varlen=weight_varlen,
        bias_varlen=bias_varlen,
        cu_seqlens=cu_seqlens,
        g_batch=g_batch,
        g_varlen=g_varlen,
    )


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------


def check_correctness(
    batch, dim, seqlen, width, dtype, has_bias, activation, device="cuda"
):
    data = make_shared_data(batch, dim, seqlen, width, dtype, has_bias, device)

    # --- forward ---
    out_batch = causal_conv1d_fn(
        data["x_batch"],
        data["weight_batch"],
        data["bias_batch"],
        seq_idx=data["seq_idx"],
        activation=activation,
    )
    out_varlen = causal_conv1d_varlen_fn(
        data["x_varlen"],
        data["weight_varlen"],
        data["bias_varlen"],
        cu_seqlens=data["cu_seqlens"],
        activation=activation,
    )

    # Reshape varlen output to batched layout for comparison
    out_varlen_batched = rearrange(out_varlen, "(b s) d -> b d s", b=batch)

    fwd_max_diff = (out_batch.float() - out_varlen_batched.float()).abs().max().item()
    fwd_mean_diff = (out_batch.float() - out_varlen_batched.float()).abs().mean().item()

    # --- backward ---
    out_batch.backward(data["g_batch"])
    out_varlen.backward(data["g_varlen"])

    dx_batch = data["x_batch"].grad
    dx_varlen = rearrange(data["x_varlen"].grad, "(b s) d -> b d s", b=batch)
    dx_max_diff = (dx_batch.float() - dx_varlen.float()).abs().max().item()
    dx_mean_diff = (dx_batch.float() - dx_varlen.float()).abs().mean().item()

    dw_batch = data["weight_batch"].grad
    dw_varlen = data["weight_varlen"].grad
    dw_max_diff = (dw_batch.float() - dw_varlen.float()).abs().max().item()
    dw_mean_diff = (dw_batch.float() - dw_varlen.float()).abs().mean().item()

    db_max_diff = db_mean_diff = 0.0
    if has_bias:
        db_batch = data["bias_batch"].grad
        db_varlen = data["bias_varlen"].grad
        db_max_diff = (db_batch.float() - db_varlen.float()).abs().max().item()
        db_mean_diff = (db_batch.float() - db_varlen.float()).abs().mean().item()

    return dict(
        fwd_max=fwd_max_diff,
        fwd_mean=fwd_mean_diff,
        dx_max=dx_max_diff,
        dx_mean=dx_mean_diff,
        dw_max=dw_max_diff,
        dw_mean=dw_mean_diff,
        db_max=db_max_diff,
        db_mean=db_mean_diff,
    )


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def bench_timing(
    batch,
    dim,
    seqlen,
    width,
    dtype,
    has_bias,
    activation,
    device="cuda",
    warmup=25,
    rep=100,
    iters=10,
):
    data = make_shared_data(batch, dim, seqlen, width, dtype, has_bias, device)

    # --- batched ---
    def fwd_batch():
        return causal_conv1d_fn(
            data["x_batch"],
            data["weight_batch"],
            data["bias_batch"],
            seq_idx=data["seq_idx"],
            activation=activation,
        )

    def fwdbwd_batch():
        out = causal_conv1d_fn(
            data["x_batch"],
            data["weight_batch"],
            data["bias_batch"],
            seq_idx=data["seq_idx"],
            activation=activation,
        )
        out.backward(data["g_batch"], retain_graph=False)
        data["x_batch"].grad = None
        data["weight_batch"].grad = None
        if data["bias_batch"] is not None:
            data["bias_batch"].grad = None

    # --- varlen ---
    def fwd_varlen():
        return causal_conv1d_varlen_fn(
            data["x_varlen"],
            data["weight_varlen"],
            data["bias_varlen"],
            cu_seqlens=data["cu_seqlens"],
            activation=activation,
        )

    def fwdbwd_varlen():
        out = causal_conv1d_varlen_fn(
            data["x_varlen"],
            data["weight_varlen"],
            data["bias_varlen"],
            cu_seqlens=data["cu_seqlens"],
            activation=activation,
        )
        out.backward(data["g_varlen"], retain_graph=False)
        data["x_varlen"].grad = None
        data["weight_varlen"].grad = None
        if data["bias_varlen"] is not None:
            data["bias_varlen"].grad = None

    fwd_b = _do_bench(fwd_batch, warmup, rep, iters)
    fwdbwd_b = _do_bench(fwdbwd_batch, warmup, rep, iters)
    bwd_b = fwdbwd_b - fwd_b

    fwd_v = _do_bench(fwd_varlen, warmup, rep, iters)
    fwdbwd_v = _do_bench(fwdbwd_varlen, warmup, rep, iters)
    bwd_v = fwdbwd_v - fwd_v

    return dict(
        fwd_batch=fwd_b,
        bwd_batch=bwd_b,
        fwdbwd_batch=fwdbwd_b,
        fwd_varlen=fwd_v,
        bwd_varlen=bwd_v,
        fwdbwd_varlen=fwdbwd_v,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description="Benchmark + correctness: varlen vs batch causal conv1d"
    )
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--dim", type=int, default=4096)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--width", type=int, default=4, choices=[2, 3, 4])
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    p.add_argument("--no-bias", action="store_true")
    p.add_argument("--activation", choices=["none", "silu"], default="silu")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=100)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument(
        "--skip-timing", action="store_true", help="Only run correctness check"
    )
    args = p.parse_args()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    has_bias = not args.no_bias
    activation = "silu" if args.activation == "silu" else None

    total_tokens = args.batch * args.seqlen

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"batch={args.batch}  dim={args.dim}  seqlen={args.seqlen}  width={args.width}  "
        f"dtype={args.dtype}  bias={has_bias}  activation={args.activation}"
    )
    print(f"total_tokens = {total_tokens}")
    print()

    # ------------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------------
    print("=" * 72)
    print("CORRECTNESS CHECK (varlen vs batch with identical data)")
    print("=" * 72)

    diffs = check_correctness(
        args.batch, args.dim, args.seqlen, args.width, dtype, has_bias, activation
    )

    # Choose tolerance based on dtype.
    # We are comparing two *different* implementations (CUDA batched kernel vs
    # Triton varlen kernel) that compute the same mathematical operation but
    # with different summation orders, atomics, etc.  The max-abs diff can have
    # outliers while the mean diff is negligible, so we check both.
    if dtype == torch.float32:
        atol_act = 1e-3  # activation outputs
        atol_w = 2e-3  # weight/bias grads (atomics in varlen)
    elif dtype == torch.bfloat16:
        atol_act = 6.5e-2  # bf16 ULP is ~0.0078 at magnitude ~1
        atol_w = 5e-3
    else:  # fp16
        atol_act = 5e-3
        atol_w = 5e-3

    print(
        f"  Forward output  -- max diff: {diffs['fwd_max']:.6e}  mean diff: {diffs['fwd_mean']:.6e}"
    )
    print(
        f"  dx (input grad) -- max diff: {diffs['dx_max']:.6e}  mean diff: {diffs['dx_mean']:.6e}"
    )
    print(
        f"  dweight         -- max diff: {diffs['dw_max']:.6e}  mean diff: {diffs['dw_mean']:.6e}"
    )
    if has_bias:
        print(
            f"  dbias           -- max diff: {diffs['db_max']:.6e}  mean diff: {diffs['db_mean']:.6e}"
        )

    # Pass/fail
    all_ok = True
    if diffs["fwd_max"] > atol_act:
        print(
            f"  [FAIL] Forward output max diff {diffs['fwd_max']:.6e} > atol {atol_act}"
        )
        all_ok = False
    if diffs["dx_max"] > atol_act:
        print(f"  [FAIL] dx max diff {diffs['dx_max']:.6e} > atol {atol_act}")
        all_ok = False
    if diffs["dw_max"] > atol_w:
        print(f"  [FAIL] dweight max diff {diffs['dw_max']:.6e} > atol {atol_w}")
        all_ok = False
    if has_bias and diffs["db_max"] > atol_w:
        print(f"  [FAIL] dbias max diff {diffs['db_max']:.6e} > atol {atol_w}")
        all_ok = False

    if all_ok:
        print("  [PASS] All outputs match within tolerance.")
    else:
        print("  [FAIL] Some outputs exceed tolerance!")
    print()

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------
    if not args.skip_timing:
        print("=" * 72)
        print("TIMING BENCHMARK")
        print("=" * 72)

        t = bench_timing(
            args.batch,
            args.dim,
            args.seqlen,
            args.width,
            dtype,
            has_bias,
            activation,
            warmup=args.warmup,
            rep=args.rep,
            iters=args.iters,
        )

        def ratio(a, b):
            return f"{a / b:.2f}x" if b > 0 else "N/A"

        header = f"{'':32} {'fwd (ms)':>10} {'bwd (ms)':>10} {'fwd+bwd':>10}"
        print(header)
        print("-" * len(header))
        print(
            f"{'Batched CUDA (causal_conv1d_fn)':32} "
            f"{t['fwd_batch']:10.3f} {t['bwd_batch']:10.3f} {t['fwdbwd_batch']:10.3f}"
        )
        print(
            f"{'Varlen Triton (causal_conv1d_varlen_fn)':32} "
            f"{t['fwd_varlen']:10.3f} {t['bwd_varlen']:10.3f} {t['fwdbwd_varlen']:10.3f}"
        )
        print()
        print(
            f"{'Varlen / Batched ratio':32} "
            f"{ratio(t['fwd_varlen'], t['fwd_batch']):>10} "
            f"{ratio(t['bwd_varlen'], t['bwd_batch']):>10} "
            f"{ratio(t['fwdbwd_varlen'], t['fwdbwd_batch']):>10}"
        )
        print()
        print(f"Throughput (tokens/ms):")
        print(
            f"  Batched  fwd: {total_tokens / t['fwd_batch']:,.0f}   "
            f"bwd: {total_tokens / t['bwd_batch']:,.0f}   "
            f"fwd+bwd: {total_tokens / t['fwdbwd_batch']:,.0f}"
        )
        print(
            f"  Varlen   fwd: {total_tokens / t['fwd_varlen']:,.0f}   "
            f"bwd: {total_tokens / t['bwd_varlen']:,.0f}   "
            f"fwd+bwd: {total_tokens / t['fwdbwd_varlen']:,.0f}"
        )

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
