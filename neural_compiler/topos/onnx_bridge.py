"""
ONNX Bridge for Agda-generated Neural Networks

Converts JSON representation from Agda's Neural.Compile.ONNX.Serialize
to actual ONNX protobuf format.

This bridges the gap between:
- Agda: Formal specification and category-theoretic guarantees
- ONNX: Executable neural network format
"""

import json
import onnx
from onnx import helper, TensorProto, numpy_helper
from typing import Dict, List, Any


# Tensor element type mapping
ELEM_TYPE_MAP = {
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


def parse_attribute_value(attr_val: Dict[str, Any]) -> Any:
    """Parse attribute value from Agda JSON."""
    if "attr-float" in attr_val:
        return float(attr_val["attr-float"])
    elif "attr-int" in attr_val:
        return int(attr_val["attr-int"])
    elif "attr-string" in attr_val:
        return str(attr_val["attr-string"])
    elif "attr-floats" in attr_val:
        return [float(x) for x in attr_val["attr-floats"]]
    elif "attr-ints" in attr_val:
        return [int(x) for x in attr_val["attr-ints"]]
    elif "attr-strings" in attr_val:
        return [str(x) for x in attr_val["attr-strings"]]
    else:
        raise ValueError(f"Unknown attribute value type: {attr_val}")


def parse_dimension(dim: Dict[str, Any]) -> int:
    """Parse dimension from Agda JSON."""
    if "dim-value" in dim:
        return int(dim["dim-value"])
    elif "dim-param" in dim:
        # Symbolic dimension - use -1
        return -1
    else:
        raise ValueError(f"Unknown dimension type: {dim}")


def json_to_onnx(json_data: Dict[str, Any]) -> onnx.ModelProto:
    """
    Convert Agda-generated JSON to ONNX ModelProto.

    Args:
        json_data: JSON dictionary from Agda's model-to-json

    Returns:
        ONNX ModelProto ready for serialization
    """

    graph_json = json_data["graph"]

    # Parse inputs
    inputs = []
    for inp in graph_json["inputs"]:
        tensor_type = inp["type"]["tensor-type"]
        elem_type = ELEM_TYPE_MAP[tensor_type["elem-type"]]
        shape = [parse_dimension(d) for d in tensor_type["shape"]]

        inputs.append(helper.make_tensor_value_info(
            inp["name"],
            elem_type,
            shape
        ))

    # Parse outputs
    outputs = []
    for out in graph_json["outputs"]:
        tensor_type = out["type"]["tensor-type"]
        elem_type = ELEM_TYPE_MAP[tensor_type["elem-type"]]
        shape = [parse_dimension(d) for d in tensor_type["shape"]]

        outputs.append(helper.make_tensor_value_info(
            out["name"],
            elem_type,
            shape
        ))

    # Parse nodes
    nodes = []
    for node_json in graph_json["nodes"]:
        # Parse attributes
        attributes = []
        for attr in node_json["attributes"]:
            attr_name = attr["name"]
            attr_value = parse_attribute_value(attr["value"])

            if isinstance(attr_value, float):
                attributes.append(helper.make_attribute(attr_name, attr_value))
            elif isinstance(attr_value, int):
                attributes.append(helper.make_attribute(attr_name, attr_value))
            elif isinstance(attr_value, str):
                attributes.append(helper.make_attribute(attr_name, attr_value))
            elif isinstance(attr_value, list):
                if attr_value and isinstance(attr_value[0], float):
                    attributes.append(helper.make_attribute(attr_name, attr_value))
                elif attr_value and isinstance(attr_value[0], int):
                    attributes.append(helper.make_attribute(attr_name, attr_value))
                elif attr_value and isinstance(attr_value[0], str):
                    attributes.append(helper.make_attribute(attr_name, attr_value))

        # Build attribute dict with raw values, not AttributeProto
        attr_dict = {}
        for attr in attributes:
            attr_dict[attr.name] = attr

        node = helper.make_node(
            node_json["op-type"],
            node_json["inputs"],
            node_json["outputs"],
            name=node_json["name"],
            domain=node_json.get("domain", "")
        )
        # Manually add attributes
        node.attribute.extend(attributes)
        nodes.append(node)

    # Parse initializers (weight tensors)
    initializers = []
    for tensor_json in graph_json.get("initializers", []):
        # For now, create empty tensors with correct shape
        # Real weights would come from training
        elem_type = ELEM_TYPE_MAP[tensor_json["elem-type"]]
        dims = tensor_json["dims"]

        # Create dummy data (zeros)
        import numpy as np
        if elem_type == TensorProto.FLOAT:
            data = np.zeros(dims, dtype=np.float32)
        elif elem_type == TensorProto.INT64:
            data = np.zeros(dims, dtype=np.int64)
        else:
            data = np.zeros(dims)

        tensor = numpy_helper.from_array(data, tensor_json["name"])
        initializers.append(tensor)

    # Create graph
    graph = helper.make_graph(
        nodes,
        graph_json["name"],
        inputs,
        outputs,
        initializer=initializers,
        doc_string=graph_json.get("doc", "")
    )

    # Create model
    opset_imports = []
    for opset in json_data["opset-import"]:
        opset_imports.append(helper.make_opsetid(
            opset["domain"],
            opset["version"]
        ))

    model = helper.make_model(
        graph,
        producer_name=json_data["producer-name"],
        producer_version=json_data.get("producer-version", "1.0"),
        doc_string=json_data.get("doc", ""),
        opset_imports=opset_imports
    )

    model.ir_version = json_data["ir-version"]
    model.model_version = json_data.get("model-version", 1)

    return model


def validate_onnx_model(model: onnx.ModelProto) -> bool:
    """
    Validate ONNX model.

    Args:
        model: ONNX ModelProto

    Returns:
        True if valid, raises exception otherwise
    """
    try:
        onnx.checker.check_model(model)
        return True
    except Exception as e:
        print(f"ONNX validation error: {e}")
        raise


def save_onnx_model(model: onnx.ModelProto, path: str):
    """Save ONNX model to file."""
    onnx.save(model, path)
    print(f"Saved ONNX model to {path}")


def load_json_from_file(path: str) -> Dict[str, Any]:
    """Load JSON from file."""
    with open(path, 'r') as f:
        return json.load(f)
