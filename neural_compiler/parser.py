"""
Parser for Agda-exported Neural IR JSON files.
"""

import json
from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional
from enum import Enum


# ============================================================================
# Shape Types
# ============================================================================

@dataclass
class Shape:
    """Base class for tensor shapes."""
    pass


@dataclass
class ScalarShape(Shape):
    """Scalar (rank-0 tensor)."""
    pass


@dataclass
class VecShape(Shape):
    """1D vector."""
    dim: int


@dataclass
class MatShape(Shape):
    """2D matrix."""
    rows: int
    cols: int


@dataclass
class TensorShape(Shape):
    """N-dimensional tensor."""
    dims: List[int]


def parse_shape(shape_json: Dict[str, Any]) -> Shape:
    """Parse shape from JSON."""
    shape_type = shape_json["type"]

    if shape_type == "scalar":
        return ScalarShape()
    elif shape_type == "vec":
        return VecShape(dim=shape_json["dim"])
    elif shape_type == "mat":
        return MatShape(rows=shape_json["rows"], cols=shape_json["cols"])
    elif shape_type == "tensor":
        return TensorShape(dims=shape_json["dims"])
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")


# ============================================================================
# Operation Types
# ============================================================================

class ActivationType(Enum):
    """Activation function types."""
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    GELU = "gelu"
    IDENTITY = "identity"


@dataclass
class Operation:
    """Base class for operations."""
    pass


@dataclass
class LinearOp(Operation):
    """Linear (dense/fully-connected) layer."""
    in_dim: int
    out_dim: int


@dataclass
class Conv2DOp(Operation):
    """2D convolution."""
    in_channels: int
    out_channels: int
    kernel_size: int


@dataclass
class ActivationOp(Operation):
    """Activation function."""
    activation: ActivationType


@dataclass
class ForkOp(Operation):
    """Fork (merge) operation - from topos theory."""
    arity: int  # Number of inputs to merge


@dataclass
class ResidualOp(Operation):
    """Residual (skip) connection - from conservation laws."""
    pass


@dataclass
class BatchNormOp(Operation):
    """Batch normalization."""
    features: int


@dataclass
class LayerNormOp(Operation):
    """Layer normalization."""
    features: int


@dataclass
class MaxPoolOp(Operation):
    """Max pooling."""
    kernel_size: int
    stride: int


@dataclass
class AvgPoolOp(Operation):
    """Average pooling."""
    kernel_size: int
    stride: int


@dataclass
class AttentionOp(Operation):
    """Multi-head attention - from stack semantics."""
    heads: int
    d_model: int
    d_k: int
    d_v: int


def parse_operation(op_json: Dict[str, Any]) -> Operation:
    """Parse operation from JSON."""
    op_type = op_json["type"]

    if op_type == "linear":
        return LinearOp(in_dim=op_json["in_dim"], out_dim=op_json["out_dim"])
    elif op_type == "conv2d":
        return Conv2DOp(
            in_channels=op_json["in_channels"],
            out_channels=op_json["out_channels"],
            kernel_size=op_json["kernel_size"]
        )
    elif op_type == "activation":
        act_str = op_json["activation"]
        return ActivationOp(activation=ActivationType(act_str))
    elif op_type == "fork":
        return ForkOp(arity=op_json["arity"])
    elif op_type == "residual":
        return ResidualOp()
    elif op_type == "batch_norm":
        return BatchNormOp(features=op_json["features"])
    elif op_type == "layer_norm":
        return LayerNormOp(features=op_json["features"])
    elif op_type == "max_pool":
        return MaxPoolOp(kernel_size=op_json["kernel_size"], stride=op_json["stride"])
    elif op_type == "avg_pool":
        return AvgPoolOp(kernel_size=op_json["kernel_size"], stride=op_json["stride"])
    elif op_type == "attention":
        return AttentionOp(
            heads=op_json["heads"],
            d_model=op_json["d_model"],
            d_k=op_json["d_k"],
            d_v=op_json["d_v"]
        )
    else:
        raise ValueError(f"Unknown operation type: {op_type}")


# ============================================================================
# Graph Structure
# ============================================================================

@dataclass
class Vertex:
    """Computational vertex in the neural graph."""
    id: int
    op: Operation
    input_shapes: List[Shape]
    output_shape: Shape


@dataclass
class Edge:
    """Directed edge between vertices."""
    source: int
    target: int
    shape: Shape


def parse_vertex(vertex_json: Dict[str, Any]) -> Vertex:
    """Parse vertex from JSON."""
    return Vertex(
        id=vertex_json["id"],
        op=parse_operation(vertex_json["op"]),
        input_shapes=[parse_shape(s) for s in vertex_json["input_shapes"]],
        output_shape=parse_shape(vertex_json["output_shape"])
    )


def parse_edge(edge_json: Dict[str, Any]) -> Edge:
    """Parse edge from JSON."""
    return Edge(
        source=edge_json["source"],
        target=edge_json["target"],
        shape=parse_shape(edge_json["shape"])
    )


# ============================================================================
# Properties and Resources
# ============================================================================

@dataclass
class Property:
    """Verified property from Agda."""
    name: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ResourceConstraints:
    """Resource bounds."""
    max_flops: int
    max_memory: int
    max_latency: int
    sparsity: int


def parse_property(prop: Union[str, Dict[str, Any]]) -> Property:
    """Parse property from JSON."""
    if isinstance(prop, str):
        return Property(name=prop)
    else:
        return Property(name=prop["type"], metadata=prop)


def parse_resources(resources_json: Dict[str, Any]) -> ResourceConstraints:
    """Parse resource constraints from JSON."""
    return ResourceConstraints(
        max_flops=resources_json["max_flops"],
        max_memory=resources_json["max_memory"],
        max_latency=resources_json["max_latency"],
        sparsity=resources_json["sparsity"]
    )


# ============================================================================
# Complete IR
# ============================================================================

@dataclass
class NeuralIR:
    """Complete intermediate representation of a neural architecture."""
    name: str
    vertices: List[Vertex]
    edges: List[Edge]
    inputs: List[int]  # Input vertex IDs
    outputs: List[int]  # Output vertex IDs
    properties: List[Property]
    resources: ResourceConstraints

    def get_vertex(self, vertex_id: int) -> Optional[Vertex]:
        """Get vertex by ID."""
        for v in self.vertices:
            if v.id == vertex_id:
                return v
        return None

    def topological_sort(self) -> List[Vertex]:
        """Return vertices in topological order."""
        # Build adjacency list
        adj = {v.id: [] for v in self.vertices}
        in_degree = {v.id: 0 for v in self.vertices}

        for e in self.edges:
            adj[e.source].append(e.target)
            in_degree[e.target] += 1

        # Kahn's algorithm
        queue = [vid for vid in in_degree if in_degree[vid] == 0]
        result = []

        while queue:
            vid = queue.pop(0)
            vertex = self.get_vertex(vid)
            if vertex:
                result.append(vertex)

            for neighbor in adj[vid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.vertices):
            raise ValueError("Graph has cycles!")

        return result


def parse_ir(json_path: str) -> NeuralIR:
    """Parse Neural IR from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    return NeuralIR(
        name=data["name"],
        vertices=[parse_vertex(v) for v in data["vertices"]],
        edges=[parse_edge(e) for e in data["edges"]],
        inputs=data["inputs"],
        outputs=data["outputs"],
        properties=[parse_property(p) for p in data["properties"]],
        resources=parse_resources(data["resources"])
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Parse MLP
    # ir = parse_ir("mlp.json")
    # print(f"Architecture: {ir.name}")
    # print(f"Vertices: {len(ir.vertices)}")
    # print(f"Edges: {len(ir.edges)}")
    # print(f"Properties: {[p.name for p in ir.properties]}")
    pass
