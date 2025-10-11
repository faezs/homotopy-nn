# ONNX Export Examples

This directory demonstrates the complete workflow for exporting Agda `DirectedGraph` to executable ONNX models.

## Architecture Overview

```
┌─────────────────┐
│  Agda           │
│  DirectedGraph  │  Categorical representation
│  (Type theory)  │  (Section 1.1 of paper)
└────────┬────────┘
         │ export-to-onnx
         ↓
┌─────────────────┐
│  Agda ONNX IR   │  In-memory ONNX types
│  ModelProto     │  (Neural.Compile.ONNX)
└────────┬────────┘
         │ Serialize to JSON
         ↓
┌─────────────────┐
│  JSON file      │  Intermediate format
│  model.json     │  (Human-readable)
└────────┬────────┘
         │ tools/onnx_bridge.py
         ↓
┌─────────────────┐
│  ONNX Protobuf  │  Executable format
│  model.onnx     │  (Industry standard)
└────────┬────────┘
         │ ONNX Runtime / PyTorch / TensorFlow
         ↓
┌─────────────────┐
│  Execution      │  Inference results
│  Results        │
└─────────────────┘
```

## Installation

### 1. Python Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install ONNX and runtime
pip install onnx onnxruntime numpy

# Optional: Install visualization tools
pip install netron  # For visualizing ONNX graphs
```

### 2. Verify Installation

```bash
python3 -c "import onnx; print(f'ONNX version: {onnx.__version__}')"
python3 -c "import onnxruntime; print(f'ONNX Runtime version: {onnxruntime.__version__}')"
```

## Examples

### Example 1: Simple MLP (784→256→10)

**Agda Definition**: `simple-mlp-export.agda`

A simple feedforward network for MNIST classification:
- Input: 784 neurons (28×28 flattened images)
- Hidden: 256 neurons with ReLU activation
- Output: 10 neurons with Softmax (digit classes)

**Graph Structure**:
```
Vertices: 3 (input, hidden, output)
Edges: 2 (input→hidden, hidden→output)
Architecture: Classical (no convergence points)
```

**Type-check Agda**:
```bash
agda --library-file=./libraries examples/simple-mlp-export.agda
```

**Convert to ONNX**:
```bash
# Check validity only
python3 tools/onnx_bridge.py --input examples/simple-mlp.json --check-only

# Save to .onnx file
python3 tools/onnx_bridge.py --input examples/simple-mlp.json --output examples/simple-mlp.onnx

# Execute with dummy data
python3 tools/onnx_bridge.py --input examples/simple-mlp.json --execute
```

**Visualize** (optional):
```bash
netron examples/simple-mlp.onnx
# Opens web browser with interactive graph visualization
```

## Workflow Details

### Step 1: Define DirectedGraph in Agda

```agda
postulate simple-mlp-graph : DirectedGraph

-- Specify structure
postulate
  simple-mlp-vertices : vertices simple-mlp-graph ≡ 3
  simple-mlp-edges : edges simple-mlp-graph ≡ 2
```

### Step 2: Annotate with ONNX Metadata

```agda
simple-mlp-annotations : GraphAnnotations simple-mlp-graph
simple-mlp-annotations = record
  { vertex-op-type = λ v → ...    -- "Input", "Gemm", "Softmax"
  ; edge-shape = λ e → ...         -- [784, 256], [256, 10]
  ; edge-elem-type = λ _ → FLOAT
  ; graph-inputs = fzero ∷ []
  ; graph-outputs = ...
  ; ...
  }
```

### Step 3: Export to ONNX

```agda
simple-mlp-onnx : ModelProto
simple-mlp-onnx = export-to-onnx simple-mlp-graph simple-mlp-annotations
```

### Step 4: Serialize to JSON

Currently manual (TODO: automate with Agda reflection or Haskell MCP).

See `simple-mlp.json` for the JSON schema.

### Step 5: Convert to ONNX Protobuf

```bash
python3 tools/onnx_bridge.py \
  --input examples/simple-mlp.json \
  --output examples/simple-mlp.onnx
```

This creates a binary `.onnx` file that can be loaded by:
- ONNX Runtime (C++, Python, JavaScript, ...)
- PyTorch (`torch.onnx`)
- TensorFlow (`tf2onnx`)
- Any ONNX-compatible runtime

### Step 6: Execute

```bash
python3 tools/onnx_bridge.py \
  --input examples/simple-mlp.json \
  --execute
```

Output:
```
Model inputs: ['input']
Model outputs: ['probabilities']

Creating dummy inputs (random data)...
  input: shape=[1, 784], dtype=float32

Running inference...

Results:
  probabilities: shape=(1, 10), dtype=float32
    min=0.0856, max=0.1124, mean=0.1000
```

## JSON Schema

The JSON format matches the Agda ONNX IR types from `Neural.Compile.ONNX`:

```json
{
  "ir-version": 9,
  "opset-import": [{"domain": "", "version": 17}],
  "producer-name": "homotopy-nn",
  "graph": {
    "name": "model-name",
    "nodes": [
      {
        "op-type": "Conv" | "Add" | "Relu" | "Gemm" | ...,
        "inputs": ["tensor_name_1", ...],
        "outputs": ["tensor_name_2", ...],
        "attributes": [
          {"name": "attr_name", "value": {"attr-int": 3}}
        ],
        "name": "node_name",
        "domain": ""
      }
    ],
    "inputs": [
      {
        "name": "input_tensor",
        "type": {
          "tensor-type": {
            "elem-type": "FLOAT",
            "shape": [{"dim-value": 224}, {"dim-param": "batch"}]
          }
        },
        "doc": "description"
      }
    ],
    "outputs": [...],
    "initializers": [
      {
        "name": "weight",
        "elem-type": "FLOAT",
        "dims": [256, 512]
      }
    ]
  }
}
```

### Supported Element Types

- `FLOAT` (32-bit float)
- `DOUBLE` (64-bit float)
- `INT32`, `INT64`
- `UINT8`, `UINT16`, `UINT32`, `UINT64`
- `FLOAT16`, `BFLOAT16`
- `BOOL`, `STRING`
- `COMPLEX64`, `COMPLEX128`

### Dimension Types

- Fixed: `{"dim-value": 224}`
- Dynamic: `{"dim-param": "batch_size"}`

### Attribute Types

- `{"attr-int": 3}`
- `{"attr-float": 1.5}`
- `{"attr-string": "SAME_UPPER"}`
- `{"attr-ints": [1, 2, 3]}`
- `{"attr-floats": [0.1, 0.2, 0.3]}`
- `{"attr-strings": ["a", "b", "c"]}`

## Connection to Paper (Belfiore & Bennequin 2022)

### Section 1.1: Oriented Graphs

The examples demonstrate how categorical oriented graphs map to computational graphs:

| Paper (Γ)                  | ONNX                        | Agda Type                |
|----------------------------|-----------------------------|--------------------------|
| Vertices (layers Lₖ)      | Subgraphs of nodes          | `Fin (vertices G)`       |
| Edges (connections)        | Tensor flow (names)         | `Fin (edges G)`          |
| source : E → V             | Node inputs                 | `source : Fin E → Fin V` |
| target : E → V             | Node outputs                | `target : Fin E → Fin V` |
| Acyclic property           | DAG constraint              | Provable in Agda         |

### Section 1.3: Fork Construction

Multi-input nodes (convergence points) correspond to fork-star vertices:

| Paper                      | ONNX                        | Example                  |
|----------------------------|-----------------------------|--------------------------|
| fork-star a★               | Multi-input node            | `Add`, `Concat`          |
| fork-tang A                | Output tensor               | Single tensor name       |
| handle a                   | Next consuming node         | Node with tensor input   |

## Troubleshooting

### Error: "Module not found"

Make sure you've installed the Python dependencies:
```bash
pip install onnx onnxruntime numpy
```

### Error: "Model validation failed"

Common issues:
1. **Tensor shape mismatch**: Output of one node doesn't match input of next
2. **Missing initializers**: Referenced weights not provided
3. **Invalid operator**: Check ONNX operator set version compatibility

Run with `--check-only` to see detailed error messages.

### Error: "Termination checking failed"

This is an Agda error, not Python. Make sure you're using the latest version of `Export.agda` with fixed edge filtering functions.

## Future Work

### Planned Features

1. **Automatic JSON serialization**: Use Agda reflection or Haskell MCP
2. **Weight export**: Extract trained weights from Agda functors
3. **Fork construction**: Map fork-star vertices to ONNX Add/Concat
4. **Reverse import**: Parse ONNX → DirectedGraph
5. **Correctness proofs**: Roundtrip property preservation
6. **ResNet example**: Skip connections with convergence
7. **Attention mechanism**: Multi-head attention architecture

### Example TODO List

- [ ] Simple chain network (MLP) ✓ Done
- [ ] ResNet block with skip connection
- [ ] Attention mechanism (Transformer layer)
- [ ] CNN (convolutional layers)
- [ ] GAN (dual networks with adversarial training)
- [ ] Parse real PyTorch model → Agda
- [ ] Prove graph properties (acyclicity, SSA)

## References

- **ONNX Specification**: https://onnx.ai/onnx/repo-docs/IR.html
- **ONNX Operators**: https://onnx.ai/onnx/operators/
- **Paper**: Belfiore & Bennequin (2022), "The geometrical structure of feedforward neural networks"
- **Agda Implementation**: `src/Neural/Compile/ONNX/Export.agda`
