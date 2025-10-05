#!/usr/bin/env python3
"""
Trace a small model or dataset to produce:
  - G.json: graph with nodes/edges (directed)
  - codes.npz: boolean activation codes (samples x neurons) and metadata

Supports two sources of connectivity:
  1) Weight matrix (JSON/NPZ) with threshold on |w_ij|
  2) Empirical mutual information (from provided activations) with threshold

CLI examples:
  python scripts/trace_model.py --weights weights.json --edge-threshold 0.2
  python scripts/trace_model.py --dataset data.npz --mi-threshold 0.02
  python scripts/trace_model.py --dataset data.csv --edge-threshold 0.2 --outdir out

Notes:
  - Dataset may be CSV (rows=samples) or NPZ with key 'X'
  - Codes are produced by thresholding per-neuron activations at median by default
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
from sklearn.metrics import mutual_info_score


@dataclass
class TraceConfig:
    weights_path: Optional[str]
    dataset_path: Optional[str]
    outdir: str
    edge_threshold: float
    mi_threshold: Optional[float]
    code_threshold: str
    seed: int


def load_weights(path: str) -> np.ndarray:
    if path.endswith(".json"):
        with open(path) as f:
            obj = json.load(f)
        arr = np.array(obj["weights"], dtype=float)
    elif path.endswith(".npz"):
        obj = np.load(path)
        key = "weights" if "weights" in obj else list(obj.keys())[0]
        arr = obj[key]
    else:
        raise ValueError(f"Unsupported weights format: {path}")
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("weights must be a square 2D matrix")
    return arr


def load_dataset(path: str) -> np.ndarray:
    if path.endswith(".csv"):
        X = np.loadtxt(path, delimiter=",")
    elif path.endswith(".npz"):
        obj = np.load(path)
        key = "X" if "X" in obj else list(obj.keys())[0]
        X = obj[key]
    else:
        raise ValueError(f"Unsupported dataset format: {path}")
    if X.ndim != 2:
        raise ValueError("dataset must be 2D (samples x neurons)")
    return X


def binarize_codes(X: np.ndarray, method: str = "median") -> np.ndarray:
    if method == "median":
        thr = np.median(X, axis=0, keepdims=True)
        return (X >= thr).astype(np.int8)
    elif method.startswith("fixed:"):
        val = float(method.split(":", 1)[1])
        return (X >= val).astype(np.int8)
    else:
        raise ValueError(f"Unknown code threshold method: {method}")


def edges_from_weights(W: np.ndarray, thr: float) -> List[Tuple[int, int]]:
    idx = np.argwhere(np.abs(W) > thr)
    # Remove self-loops
    idx = idx[idx[:, 0] != idx[:, 1]]
    return [(int(i), int(j)) for i, j in idx]


def edges_from_mi(codes: np.ndarray, thr: float) -> List[Tuple[int, int]]:
    n = codes.shape[1]
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        xi = codes[:, i]
        for j in range(n):
            if i == j:
                continue
            xj = codes[:, j]
            mi = float(mutual_info_score(xi, xj))
            if mi > thr:
                edges.append((i, j))
    return edges


def write_graph(outdir: str, n: int, edges: Iterable[Tuple[int, int]]) -> str:
    nodes = [f"n{i}" for i in range(n)]
    G = {"nodes": nodes, "edges": [(nodes[i], nodes[j]) for i, j in edges], "directed": True}
    os.makedirs(outdir, exist_ok=True)
    outp = os.path.join(outdir, "G.json")
    with open(outp, "w") as f:
        json.dump(G, f)
    return outp


def write_codes(outdir: str, codes: np.ndarray, meta: dict) -> str:
    os.makedirs(outdir, exist_ok=True)
    outp = os.path.join(outdir, "codes.npz")
    np.savez_compressed(outp, codes=codes, meta=json.dumps(meta))
    return outp


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Trace model/dataset to graph and codes")
    p.add_argument("--weights", dest="weights_path", default=None, help="Path to weights (JSON with {weights:[[...]]} or NPZ)")
    p.add_argument("--dataset", dest="dataset_path", default=None, help="Path to dataset (CSV or NPZ with X)")
    p.add_argument("--outdir", default="out", help="Output directory")
    p.add_argument("--edge-threshold", type=float, default=0.2, help="|w_ij| threshold for edges from weights")
    p.add_argument("--mi-threshold", type=float, default=None, help="Mutual information threshold for edges from codes")
    p.add_argument("--code-threshold", default="median", help="Code thresholding: 'median' or 'fixed:<val>'")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    cfg = TraceConfig(
        weights_path=args.weights_path,
        dataset_path=args.dataset_path,
        outdir=args.outdir,
        edge_threshold=args.edge_threshold,
        mi_threshold=args.mi_threshold,
        code_threshold=args.code_threshold,
        seed=args.seed,
    )

    np.random.seed(cfg.seed)

    n_neurons: Optional[int] = None
    edges: List[Tuple[int, int]] = []
    codes: Optional[np.ndarray] = None

    if cfg.weights_path:
        W = load_weights(cfg.weights_path)
        n_neurons = W.shape[0]
        edges += edges_from_weights(W, cfg.edge_threshold)

    if cfg.dataset_path:
        X = load_dataset(cfg.dataset_path)
        if n_neurons is None:
            n_neurons = X.shape[1]
        elif X.shape[1] != n_neurons:
            raise ValueError("dataset neuron count does not match weights")
        codes = binarize_codes(X, cfg.code_threshold)
        if cfg.mi_threshold is not None:
            edges += edges_from_mi(codes, cfg.mi_threshold)

    if n_neurons is None:
        # No inputs provided; generate a tiny toy DAG
        n_neurons = 5
        edges = [(0, 2), (1, 3), (2, 4), (3, 4)]
        codes = (np.random.rand(32, n_neurons) > 0.5).astype(np.int8)

    graph_path = write_graph(cfg.outdir, n_neurons, edges)
    meta = {"edge_threshold": cfg.edge_threshold, "mi_threshold": cfg.mi_threshold}
    codes_path = write_codes(cfg.outdir, codes if codes is not None else np.zeros((0, n_neurons), dtype=np.int8), meta)

    print(f"Wrote {graph_path}")
    print(f"Wrote {codes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
