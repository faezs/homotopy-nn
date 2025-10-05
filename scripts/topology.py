#!/usr/bin/env python3
"""
Compute homology of the clique complex built from an (un)directed graph.

Primary path uses GUDHI for accurate Betti numbers. Fallback computes:
  - beta0 via connected components
  - beta1 via Euler characteristic of 2-skeleton (E - V + C - #triangles)

CLI examples:
  python scripts/topology.py --graph out/G.json --out out/homology.json
  python scripts/topology.py --graph out/G.json --max-dim 3 --use-gudhi
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Set, Tuple

import networkx as nx

try:
    import gudhi  # type: ignore
    HAS_GUDHI = True
except Exception:
    HAS_GUDHI = False


def load_graph(path: str) -> Tuple[List[str], List[Tuple[str, str]], bool]:
    with open(path) as f:
        G = json.load(f)
    nodes = list(G["nodes"])  # preserve order
    edges = [(u, v) for (u, v) in G["edges"]]
    directed = bool(G.get("directed", False))
    return nodes, edges, directed


def build_undirected(nodes: List[str], edges: Iterable[Tuple[str, str]], directed: bool) -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(nodes)
    if directed:
        # use undirected support
        for u, v in edges:
            if u == v:
                continue
            H.add_edge(u, v)
    else:
        H.add_edges_from([(u, v) for (u, v) in edges if u != v])
    return H


def betti_gudhi(H: nx.Graph, max_dim: int) -> Dict[int, int]:
    """Compute Betti numbers using GUDHI. Map node labels to ints.

    GUDHI's SimplexTree expects integer vertex ids. We map node labels to
    consecutive integers (preserving insertion order) for the construction
    and only use those ints internally.
    """
    # Map nodes to integer ids for GUDHI
    nodes_in_order = list(H.nodes)
    id_of = {u: i for i, u in enumerate(nodes_in_order)}

    # Build clique complex via maximal cliques
    st = gudhi.SimplexTree()
    # Add vertices
    for u in nodes_in_order:
        st.insert([id_of[u]], filtration=0.0)
    # Add higher cliques (insert maximal cliques; expansion fills faces)
    for clique in nx.find_cliques(H):
        if len(clique) >= 2:
            st.insert([id_of[v] for v in clique], filtration=0.0)
    st.expansion(max_dim)
    # GUDHI requires computing persistence (even if all filtrations are 0)
    # before requesting Betti numbers.
    try:
        st.compute_persistence()
    except AttributeError:
        # Older API
        _ = st.persistence()
    bettis = st.betti_numbers()
    # bettis list length may be > max_dim+1; trim
    out: Dict[int, int] = {}
    for i in range(0, min(len(bettis), max_dim + 1)):
        out[i] = int(bettis[i])
    return out


def betti_fallback(H: nx.Graph) -> Dict[int, int]:
    # beta0: number of connected components
    beta0 = nx.number_connected_components(H)
    # beta1 via Euler characteristic of 2-skeleton: beta1 = E - V + C - F
    V = H.number_of_nodes()
    E = H.number_of_edges()
    # number of triangles (3-cliques)
    F = sum(nx.triangles(H).values()) // 3
    beta1 = max(E - V + beta0 - F, 0)
    return {0: int(beta0), 1: int(beta1)}


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compute Betti numbers of clique complex")
    p.add_argument("--graph", default="out/G.json")
    p.add_argument("--out", default="out/homology.json")
    p.add_argument("--max-dim", type=int, default=2)
    p.add_argument("--use-gudhi", action="store_true", help="Force use of GUDHI (errors if unavailable)")
    args = p.parse_args(argv)

    nodes, edges, directed = load_graph(args.graph)
    H = build_undirected(nodes, edges, directed)

    if args.use_gudhi and not HAS_GUDHI:
        raise SystemExit("--use-gudhi requested but gudhi is not available in this environment")

    if HAS_GUDHI:
        beta = betti_gudhi(H, args.max_dim)
    else:
        beta = betti_fallback(H)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"beta": {str(k): v for k, v in beta.items()}}, f)

    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
