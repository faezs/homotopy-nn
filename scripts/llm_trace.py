#!/usr/bin/env python3
"""
LLM structural/dynamic tracer for large HF models (e.g., Gemma).

Goals
- Produce scalable graph summaries from a large model without materializing
  full neuron graphs in memory.
- Two complementary graphs:
  1) Structural graph from weight matrices (top‑K incoming edges per sampled
     output unit), recorded sparsely.
  2) Effective graph from runtime attention (averaged per head/layer over a
     prompt set), thresholded and stored sparsely.

Requirements
- Run inside the Nix devshell (nix develop) which provides torch/transformers.
- For offline use, pass a local model path via --model-id.

Outputs (under --outdir)
- meta.json: configuration and shapes
- structural/
    layer_<i>__<name>.npz: CSR sparse top‑K edges for that Linear module
- effective/
    attn_layer_<i>.npz: CSR sparse adjacency between token positions (avg over prompts)
- summaries.json: per‑layer stats (edge densities, norms, thresholds)

Usage examples
- python scripts/llm_trace.py --model-id /path/to/local/gemma-2b --device cpu --sample-out 0.05 --topk 8
- python scripts/llm_trace.py --model-id google/gemma-2-2b --device cuda --dtype float16 --prompts prompts.txt --max-prompts 64 --attn-threshold 0.02

Notes
- Designed to degrade gracefully on CPU with sampling; for full coverage, run
  on CUDA with fp16 and limit prompt length/max prompts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    HAS_HF = True
except Exception:
    HAS_HF = False


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


@dataclass
class Config:
    model_id: str
    device: str
    dtype: str
    outdir: str
    topk: int
    sample_out: float
    prompts: Optional[str]
    max_prompts: int
    max_tokens: int
    attn_threshold: float
    attn_max_batches: int


def torch_dtype(name: str):
    name = name.lower()
    if name in ("float32", "fp32"):
        return torch.float32
    if name in ("float16", "fp16", "half"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def sparse_topk_from_linear(W: torch.Tensor, k: int, sample_out: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Return CSR arrays (data, indices, indptr) for top‑K per sampled output row."""
    out_dim, in_dim = W.shape
    # choose rows to sample
    n_rows = max(1, int(math.ceil(out_dim * sample_out)))
    row_idx = torch.linspace(0, out_dim - 1, n_rows).round().long().unique()
    # compute top‑K indices by absolute weight
    Wabs = W.abs()
    topk_vals, topk_idx = torch.topk(Wabs[row_idx], k=min(k, in_dim), dim=1)
    # build CSR
    data = topk_vals.detach().cpu().numpy().astype(np.float32).ravel()
    indices = topk_idx.detach().cpu().numpy().astype(np.int32).ravel()
    indptr = np.arange(0, data.size + 1, k, dtype=np.int32)
    return data, indices, indptr, int(n_rows), int(in_dim)


def save_csr_npz(path: str, data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, shape: Tuple[int, int], meta: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, data=data, indices=indices, indptr=indptr, shape=np.array(shape), meta=json.dumps(meta))


def iter_linear_modules(model: nn.Module):
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            yield name, mod


def trace_structural(model: nn.Module, cfg: Config, device: torch.device, dtype_torch) -> List[Dict]:
    stats: List[Dict] = []
    for idx, (name, lin) in enumerate(iter_linear_modules(model)):
        W = lin.weight.data.to(device)
        W = W.to(dtype_torch)
        data, indices, indptr, n_rows, in_dim = sparse_topk_from_linear(W, cfg.topk, cfg.sample_out)
        path = os.path.join(cfg.outdir, "structural", f"layer_{idx:03d}__{name.replace('.', '_')}.npz")
        save_csr_npz(path, data, indices, indptr, (n_rows, in_dim), {"layer": idx, "name": name, "topk": cfg.topk, "sample_out": cfg.sample_out})
        stats.append({"layer": idx, "name": name, "rows": n_rows, "cols": in_dim, "nnz": int(data.size)})
    return stats


def load_prompts(path: str, limit: int) -> List[str]:
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines[:limit]


def trace_effective(model, tokenizer, cfg: Config, device: torch.device, dtype_torch) -> List[Dict]:
    """Collect average attention matrices per layer over a prompt set, threshold into sparse CSR."""
    model.eval()
    layers = []
    attn_accum: Dict[int, torch.Tensor] = {}

    # hook to intercept attention probs; assume model exposes attn probs via outputs.attentions when configured
    # we’ll instead use generate a forward pass with output_attentions=True
    prompts = load_prompts(cfg.prompts, cfg.max_prompts) if cfg.prompts else ["Hello world"]
    max_batches = min(cfg.attn_max_batches, len(prompts))

    with torch.no_grad():
        for i in range(max_batches):
            batch = prompts[i:i+1]
            toks = tokenizer(batch, return_tensors="pt", padding=False, truncation=True, max_length=cfg.max_tokens)
            toks = {k: v.to(device) for k, v in toks.items()}
            out = model(**toks, output_attentions=True)
            atts = out.attentions  # tuple(num_layers) of (bsz, num_heads, seq, seq)
            if atts is None:
                break
            for li, A in enumerate(atts):
                # average over heads and batch
                A = A.to(dtype_torch).mean(dim=(0, 1))  # (seq, seq)
                if li not in attn_accum:
                    attn_accum[li] = A.clone()
                else:
                    attn_accum[li] += A

    stats: List[Dict] = []
    for li, A in attn_accum.items():
        A = (A / max_batches).detach().cpu()
        seq = A.shape[0]
        # threshold to build sparse adjacency
        mask = (A > cfg.attn_threshold) & (~torch.eye(seq, dtype=torch.bool))
        idx = mask.nonzero(as_tuple=False)
        data = A[mask].numpy().astype(np.float32)
        # build CSR by rows
        if idx.numel() == 0:
            indices = np.array([], dtype=np.int32)
            indptr = np.zeros(seq + 1, dtype=np.int32)
        else:
            idx_np = idx.numpy()
            order = np.lexsort((idx_np[:, 1], idx_np[:, 0]))
            idx_np = idx_np[order]
            data = data[order]
            indices = idx_np[:, 1].astype(np.int32)
            indptr = np.zeros(seq + 1, dtype=np.int32)
            # count per row
            rows, counts = np.unique(idx_np[:, 0], return_counts=True)
            indptr[rows + 1] = counts
            indptr = np.cumsum(indptr, dtype=np.int32)
        path = os.path.join(cfg.outdir, "effective", f"attn_layer_{li:03d}.npz")
        save_csr_npz(path, data, indices, indptr, (seq, seq), {"layer": li, "attn_threshold": cfg.attn_threshold})
        stats.append({"layer": li, "seq": int(seq), "nnz": int(data.size), "density": float(data.size) / max(1, seq * seq)})
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    require(HAS_TORCH, "PyTorch not available; enter nix devshell")
    require(HAS_HF, "transformers not available; enter nix devshell")

    p = argparse.ArgumentParser(description="Trace HF LLM to structural/effective sparse graphs")
    p.add_argument("--model-id", required=True, help="HF model id or local path (e.g., google/gemma-2-2b)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"]) 
    p.add_argument("--dtype", default="float16", help="float32|float16|bfloat16")
    p.add_argument("--outdir", default="out_llm")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--sample-out", type=float, default=0.05, dest="sample_out")
    p.add_argument("--prompts", default=None, help="Path to prompts.txt (one per line)")
    p.add_argument("--max-prompts", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--attn-threshold", type=float, default=0.02)
    p.add_argument("--attn-max-batches", type=int, default=32)
    args = p.parse_args(argv)

    cfg = Config(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        outdir=args.outdir,
        topk=args.topk,
        sample_out=args.sample_out,
        prompts=args.prompts,
        max_prompts=args.max_prompts,
        max_tokens=args.max_tokens,
        attn_threshold=args.attn_threshold,
        attn_max_batches=args.attn_max_batches,
    )

    os.makedirs(cfg.outdir, exist_ok=True)
    device = torch.device(cfg.device)
    dtype_t = torch_dtype(cfg.dtype)

    print(f"Loading {cfg.model_id} on {cfg.device} ({cfg.dtype})…")
    tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=dtype_t, low_cpu_mem_usage=True)
    model.to(device)

    meta = {
        "model_id": cfg.model_id,
        "device": cfg.device,
        "dtype": cfg.dtype,
        "topk": cfg.topk,
        "sample_out": cfg.sample_out,
        "attn_threshold": cfg.attn_threshold,
        "max_prompts": cfg.max_prompts,
        "max_tokens": cfg.max_tokens,
    }
    with open(os.path.join(cfg.outdir, "meta.json"), "w") as f:
        json.dump(meta, f)

    print("Tracing structural sparsity from Linear weights…")
    sstats = trace_structural(model, cfg, device, dtype_t)

    print("Tracing effective attention adjacency…")
    estats = trace_effective(model, tok, cfg, device, dtype_t)

    with open(os.path.join(cfg.outdir, "summaries.json"), "w") as f:
        json.dump({"structural": sstats, "effective": estats}, f)

    print(f"Done. Wrote outputs to {cfg.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

