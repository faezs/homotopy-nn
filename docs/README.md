# Blog Post: Neural Networks Are Functors

This directory contains the static site for the blog post on categorical foundations of neural networks.

## Structure

```
docs/
├── index.html           # Main blog post
├── css/
│   └── style.css       # Styling
├── images/             # Generated SVG diagrams
├── diagrams/           # Haskell source for diagram generation
│   ├── NetworkDiagrams.hs
│   ├── network-diagrams.cabal
│   └── Makefile
├── _config.yml         # GitHub Pages config
└── README.md           # This file
```

## Generating Diagrams

The network diagrams are generated using Haskell's diagrams library:

```bash
cd diagrams
make
```

This creates SVG files in `images/`:
- `mlp.svg` - Simple MLP chain
- `convergent.svg` - Convergent network
- `convergent_fork.svg` - Fork construction
- `convergent_poset.svg` - Resulting poset
- `complex.svg` - Complex multi-path network

## Deployment

GitHub Pages automatically serves `index.html` from this directory.

To enable GitHub Pages:
1. Go to repository Settings → Pages
2. Source: Deploy from branch
3. Branch: `claude/nix-develop-setup-01GUiQF2cgjkUAZLphdjCLdA` (or main after merge)
4. Folder: `/docs`
5. Save

The site will be available at: `https://<username>.github.io/homotopy-nn/`

## Local Testing

```bash
# Serve locally (requires Python)
cd docs
python3 -m http.server 8000

# Open browser to http://localhost:8000
```

## Content

The blog post covers:
1. **DirectedGraph = Functor ·⇉· FinSets** - The fundamental definition
2. **Network Summing Functors** - Categorical composition and conservation laws
3. **Semiring Homomorphisms** - Evaluation semantics
4. **Fork Construction** - Handling convergent layers
5. **DNN Topos** - Grothendieck toposes from architectures
6. **HIT Technique** - Avoiding K axiom in cubical Agda

Written in the style of the README: precise definitions with clear conceptual explanations.
