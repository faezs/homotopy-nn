#!/usr/bin/env python3
"""
ONNX Bridge: Serialize Agda DirectedGraph → ONNX protobuf

This module bridges the gap between Agda's typed representation of ONNX
and the actual ONNX protobuf format that can be executed by runtimes.

Workflow:
1. Agda exports DirectedGraph → ONNX ModelProto (in-memory Agda representation)
2. Serialize Agda ONNX types to JSON
3. This script converts JSON → ONNX protobuf
4. Save as .onnx file
5. Execute with ONNX Runtime
6. Verify correctness

Usage:
    python tools/onnx_bridge.py --input model.json --output model.onnx
    python tools/onnx_bridge.py --input model.json --execute
"""

import argparse
import json
import sys
from typing import Dict, List, Any, Optional

try:
    import onnx
    from onnx import helper, TensorProto, checker
except ImportError:
    print("Error: ONNX not installed. Install with: pip install onnx", file=sys.stderr)
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Warning: numpy not installed. Some features may not work.", file=sys.stderr)
    np = None

try:
    import onnxruntime as ort
except ImportError:
    print("Warning: onnxruntime not installed. Execution disabled.", file=sys.stderr)
    ort = None


# ============================================================================
# Type Conversions: Agda JSON → ONNX Protobuf
# ============================================================================

def agda_elem_type_to_onnx(elem_type: str) -> int:
    """
    Convert Agda TensorElementType to ONNX TensorProto.DataType enum.

    Mapping from Neural.Compile.ONNX:
    FLOAT      → 1   (32-bit float)
    UINT8      → 2
    INT8       → 3
    UINT16     → 4
    INT16      → 5
    INT32      → 6
    INT64      → 7
    STRING     → 8
    BOOL       → 9
    FLOAT16    → 10  (IEEE 754 half-precision)
    DOUBLE     → 11  (64-bit float)
    UINT32     → 12
    UINT64     → 13
    COMPLEX64  → 14
    COMPLEX128 → 15
    BFLOAT16   → 16  (Brain float16)
    """
    mapping = {
        "UNDEFINED": TensorProto.UNDEFINED,
        "FLOAT": TensorProto.FLOAT,
        "UINT8": TensorProto.UINT8,
        "INT8": TensorProto.INT8,
        "UINT16": TensorProto.UINT16,
        "INT16": TensorProto.INT16,
        "INT32": TensorProto.INT32,
        "INT64": TensorProto.INT64,
        "STRING": TensorProto.STRING,
        "BOOL": TensorProto.BOOL,
        "FLOAT16": TensorProto.FLOAT16,
        "DOUBLE": TensorProto.DOUBLE,
        "UINT32": TensorProto.UINT32,
        "UINT64": TensorProto.UINT64,
        "COMPLEX64": TensorProto.COMPLEX64,
        "COMPLEX128": TensorProto.COMPLEX128,
        "BFLOAT16": TensorProto.BFLOAT16,
    }
    return mapping.get(elem_type, TensorProto.UNDEFINED)


def agda_dimension_to_onnx(dim: Dict[str, Any]) -> Optional[int]:
    """
    Convert Agda Dimension to ONNX dimension.

    Agda representation:
    - dim-value n → Fixed size dimension n
    - dim-param "name" → Symbolic dimension (returns None for dynamic)
    """
    if "dim-value" in dim:
        return dim["dim-value"]
    elif "dim-param" in dim:
        # Symbolic dimension - ONNX represents as None or -1
        return None
    else:
        raise ValueError(f"Unknown dimension format: {dim}")


def agda_shape_to_onnx(shape: List[Dict[str, Any]]) -> List[Optional[int]]:
    """Convert Agda TensorShape to ONNX shape (list of dims)."""
    return [agda_dimension_to_onnx(dim) for dim in shape]


def agda_attribute_to_onnx(attr: Dict[str, Any]) -> helper.AttributeProto:
    """
    Convert Agda AttributeProto to ONNX AttributeProto.

    Agda representation:
    { "name": "...", "value": {"attr-int": 3} }
    { "name": "...", "value": {"attr-ints": [1, 2, 3]} }
    """
    name = attr["name"]
    value = attr["value"]

    if "attr-float" in value:
        return helper.make_attribute(name, float(value["attr-float"]))
    elif "attr-int" in value:
        return helper.make_attribute(name, int(value["attr-int"]))
    elif "attr-string" in value:
        return helper.make_attribute(name, str(value["attr-string"]))
    elif "attr-floats" in value:
        return helper.make_attribute(name, [float(x) for x in value["attr-floats"]])
    elif "attr-ints" in value:
        return helper.make_attribute(name, [int(x) for x in value["attr-ints"]])
    elif "attr-strings" in value:
        return helper.make_attribute(name, [str(x) for x in value["attr-strings"]])
    else:
        raise ValueError(f"Unknown attribute value format: {value}")


def agda_node_to_onnx(node: Dict[str, Any]) -> helper.NodeProto:
    """
    Convert Agda NodeProto to ONNX NodeProto.

    Agda representation:
    {
      "op-type": "Conv",
      "inputs": ["input", "weight", "bias"],
      "outputs": ["output"],
      "attributes": [...],
      "name": "conv1",
      "domain": ""
    }
    """
    # Convert attributes to dict of values (not AttributeProto objects)
    attrs_dict = {}
    for attr in node.get("attributes", []):
        name = attr["name"]
        value = attr["value"]

        if "attr-float" in value:
            attrs_dict[name] = float(value["attr-float"])
        elif "attr-int" in value:
            attrs_dict[name] = int(value["attr-int"])
        elif "attr-string" in value:
            attrs_dict[name] = str(value["attr-string"])
        elif "attr-floats" in value:
            attrs_dict[name] = [float(x) for x in value["attr-floats"]]
        elif "attr-ints" in value:
            attrs_dict[name] = [int(x) for x in value["attr-ints"]]
        elif "attr-strings" in value:
            attrs_dict[name] = [str(x) for x in value["attr-strings"]]

    return helper.make_node(
        node["op-type"],
        inputs=node["inputs"],
        outputs=node["outputs"],
        name=node.get("name", ""),
        domain=node.get("domain", ""),
        **attrs_dict
    )


def agda_value_info_to_onnx(value_info: Dict[str, Any]) -> helper.ValueInfoProto:
    """
    Convert Agda ValueInfoProto to ONNX ValueInfoProto.

    Agda representation:
    {
      "name": "input",
      "type": {"tensor-type": {"elem-type": "FLOAT", "shape": [...]}},
      "doc": "Input tensor"
    }
    """
    name = value_info["name"]
    type_proto = value_info["type"]

    if "tensor-type" not in type_proto:
        raise ValueError(f"Only tensor types supported, got: {type_proto}")

    tensor_type = type_proto["tensor-type"]
    elem_type = agda_elem_type_to_onnx(tensor_type["elem-type"])
    shape = agda_shape_to_onnx(tensor_type["shape"])

    return helper.make_tensor_value_info(name, elem_type, shape)


def agda_tensor_to_onnx(tensor: Dict[str, Any]) -> helper.TensorProto:
    """
    Convert Agda TensorProto to ONNX TensorProto.

    Note: Agda representation doesn't include actual data yet.
    This creates a placeholder tensor with zeros.
    """
    name = tensor["name"]
    elem_type = agda_elem_type_to_onnx(tensor["elem-type"])
    dims = tensor["dims"]

    # Create placeholder data (all zeros)
    # In a real implementation, this would come from trained weights
    if np is not None:
        data = np.zeros(dims, dtype=np.float32)
        return helper.make_tensor(name, elem_type, dims, data.flatten().tolist())
    else:
        # Fallback without numpy
        return helper.make_tensor(name, elem_type, dims, [0.0] * np.prod(dims))


def agda_graph_to_onnx(graph: Dict[str, Any]) -> helper.GraphProto:
    """
    Convert Agda GraphProto to ONNX GraphProto.

    Agda representation:
    {
      "nodes": [...],
      "name": "graph",
      "inputs": [...],
      "outputs": [...],
      "initializers": [...],
      "doc": "..."
    }
    """
    nodes = [agda_node_to_onnx(node) for node in graph["nodes"]]
    inputs = [agda_value_info_to_onnx(inp) for inp in graph["inputs"]]
    outputs = [agda_value_info_to_onnx(out) for out in graph["outputs"]]
    initializers = [agda_tensor_to_onnx(init) for init in graph.get("initializers", [])]

    return helper.make_graph(
        nodes,
        graph["name"],
        inputs,
        outputs,
        initializers
    )


def agda_model_to_onnx(model: Dict[str, Any]) -> onnx.ModelProto:
    """
    Convert Agda ModelProto to ONNX ModelProto.

    Agda representation:
    {
      "ir-version": 9,
      "opset-import": [{"domain": "", "version": 17}],
      "producer-name": "homotopy-nn",
      "producer-version": "1.0.0",
      "domain": "neural.homotopy",
      "model-version": 1,
      "doc": "...",
      "graph": {...}
    }
    """
    graph = agda_graph_to_onnx(model["graph"])

    # Create opset imports
    opset_imports = []
    for opset in model["opset-import"]:
        opset_imports.append(helper.make_opsetid(opset["domain"], opset["version"]))

    # Create model
    onnx_model = helper.make_model(
        graph,
        producer_name=model["producer-name"],
        producer_version=model.get("producer-version", "1.0.0"),
        ir_version=model["ir-version"],
        opset_imports=opset_imports,
        doc_string=model.get("doc", "")
    )

    # Set additional metadata
    onnx_model.domain = model.get("domain", "")
    onnx_model.model_version = model.get("model-version", 1)

    return onnx_model


# ============================================================================
# Main Conversion and Execution
# ============================================================================

def json_to_onnx(json_path: str, output_path: Optional[str] = None) -> onnx.ModelProto:
    """
    Convert JSON-serialized Agda ONNX model to ONNX protobuf.

    Args:
        json_path: Path to JSON file with Agda ONNX representation
        output_path: Optional path to save .onnx file

    Returns:
        ONNX ModelProto
    """
    print(f"Loading Agda ONNX JSON from {json_path}...")
    with open(json_path, 'r') as f:
        agda_model = json.load(f)

    print("Converting Agda representation to ONNX protobuf...")
    onnx_model = agda_model_to_onnx(agda_model)

    print("Checking ONNX model validity...")
    try:
        checker.check_model(onnx_model)
        print("✓ Model is valid ONNX")
    except Exception as e:
        print(f"⚠ Warning: Model validation failed: {e}", file=sys.stderr)

    if output_path:
        print(f"Saving ONNX model to {output_path}...")
        onnx.save(onnx_model, output_path)
        print(f"✓ Saved to {output_path}")

    return onnx_model


def execute_onnx(onnx_model: onnx.ModelProto, inputs: Optional[Dict[str, np.ndarray]] = None):
    """
    Execute ONNX model with ONNX Runtime.

    Args:
        onnx_model: ONNX ModelProto to execute
        inputs: Dictionary mapping input names to numpy arrays
    """
    if ort is None:
        print("Error: onnxruntime not installed. Cannot execute model.", file=sys.stderr)
        return

    if np is None:
        print("Error: numpy not installed. Cannot execute model.", file=sys.stderr)
        return

    print("\nCreating ONNX Runtime session...")
    session = ort.InferenceSession(onnx_model.SerializeToString())

    # Get input/output names
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    print(f"Model inputs: {input_names}")
    print(f"Model outputs: {output_names}")

    # Create dummy inputs if not provided
    if inputs is None:
        print("\nCreating dummy inputs (random data)...")
        inputs = {}
        for inp in session.get_inputs():
            shape = inp.shape
            # Replace dynamic dimensions with 1
            shape = [dim if isinstance(dim, int) else 1 for dim in shape]
            inputs[inp.name] = np.random.randn(*shape).astype(np.float32)
            print(f"  {inp.name}: shape={shape}, dtype=float32")

    # Run inference
    print("\nRunning inference...")
    outputs = session.run(output_names, inputs)

    print("\nResults:")
    for name, output in zip(output_names, outputs):
        print(f"  {name}: shape={output.shape}, dtype={output.dtype}")
        print(f"    min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Bridge Agda DirectedGraph to ONNX protobuf"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSON file with Agda ONNX representation"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output .onnx file path (optional)"
    )
    parser.add_argument(
        "--execute", "-e",
        action="store_true",
        help="Execute the model with ONNX Runtime"
    )
    parser.add_argument(
        "--check-only", "-c",
        action="store_true",
        help="Only check validity, don't save or execute"
    )

    args = parser.parse_args()

    # Convert JSON to ONNX
    try:
        onnx_model = json_to_onnx(
            args.input,
            output_path=args.output if not args.check_only else None
        )
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Execute if requested
    if args.execute and not args.check_only:
        try:
            execute_onnx(onnx_model)
        except Exception as e:
            print(f"Error during execution: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
