#!/usr/bin/env python3
"""
Persistent JAX Attention REPL Session

Reads JSON requests from stdin, executes attention operations compiled from
Agda 3-category formalization, returns results.

Protocol:
    Input (JSON line):
        {
            "operation": "attention_head" | "multi_head_attention" | "transformer_block",
            "config": {
                "n_heads": 8,
                "d_model": 512,
                "d_k": 64,
                "d_v": 64,
                "dropout": 0.1
            },
            "inputs": [
                {"shape": [batch, seq, dim], "data": [...]},
                {"shape": [batch, seq, seq], "data": [...]}  // optional mask
            ]
        }

    Output (JSON line):
        {
            "success": true,
            "output": {"shape": [batch, seq, dim], "data": [...]},
            "attention_weights": {"shape": [batch, seq, seq], "data": [...]}
        }

    Error (JSON line):
        {
            "success": false,
            "error": "ValueError: Invalid config ..."
        }

Usage:
    # Start session (waits for input)
    $ python3 attention_session.py

    # From Haskell:
    import System.Process
    (Just stdin, Just stdout, _, _) = createProcess (proc "python3" ["attention_session.py"])
                                                    { std_in = CreatePipe, std_out = CreatePipe }
    hPutStrLn stdin requestJSON
    responseJSON <- hGetLine stdout
"""

import sys
import json
import numpy as np

# Import JAX/Flax attention implementation
try:
    import jax
    import jax.numpy as jnp
    from jax import random
    from attention_jax_runtime import AttentionHead, MultiHeadAttention, TransformerBlock
except ImportError as e:
    print(json.dumps({"status": "error", "error": f"Failed to import JAX: {e}"}), flush=True)
    sys.exit(1)


def parse_tensor(tensor_json: dict):
    """
    Convert JSON tensor representation to JAX array.

    Args:
        tensor_json: {"shape": [batch, seq, dim], "data": [1.0, 2.0, ...]}

    Returns:
        JAX array with specified shape and data
    """
    shape = tensor_json['shape']
    data = tensor_json['data']
    return jnp.array(data, dtype=jnp.float32).reshape(shape)


def tensor_to_json(tensor) -> dict:
    """
    Convert JAX array to JSON representation.

    Args:
        tensor: JAX array

    Returns:
        {"shape": [...], "data": [...]}
    """
    return {
        'shape': list(tensor.shape),
        'data': tensor.flatten().tolist()
    }


def create_attention_module(operation: str, config: dict):
    """
    Create Flax attention module based on operation type and config.

    Args:
        operation: "attention_head", "multi_head_attention", or "transformer_block"
        config: Configuration dict with n_heads, d_model, d_k, d_v, dropout

    Returns:
        Tuple of (module_class, module_kwargs, requires_mask)
    """
    n_heads = config.get('n_heads', 8)
    d_model = config.get('d_model', 512)
    d_k = config.get('d_k', 64)
    d_v = config.get('d_v', 64)
    dropout = config.get('dropout', 0.1)

    if operation == "attention_head":
        return (AttentionHead, {
            'd_model': d_model,
            'd_k': d_k,
            'd_v': d_v
        }, True)  # returns attention weights

    elif operation == "multi_head_attention":
        return (MultiHeadAttention, {
            'n_heads': n_heads,
            'd_model': d_model
        }, False)  # doesn't return weights by default

    elif operation == "transformer_block":
        return (TransformerBlock, {
            'n_heads': n_heads,
            'd_model': d_model,
            'd_ff': d_model * 4,  # Standard FFN dimension
            'dropout_rate': dropout
        }, False)

    else:
        raise ValueError(f"Unknown operation: {operation}")


def handle_request(request: dict, rng_key) -> dict:
    """
    Process a single attention request.

    Args:
        request: {
            "operation": "...",
            "config": {...},
            "inputs": [{"shape": [...], "data": [...]}, ...]
        }
        rng_key: JAX random key for initialization

    Returns:
        {"success": true, "output": {...}, "attention_weights": {...}}
        or {"success": false, "error": "..."}
    """
    try:
        operation = request['operation']
        config = request['config']
        input_tensors = [parse_tensor(t) for t in request['inputs']]

        # Get main input (first tensor)
        x = input_tensors[0]
        batch_size, seq_len, _ = x.shape

        # Optional mask (second tensor if provided)
        mask = None
        if len(input_tensors) > 1:
            mask = input_tensors[1]

        # Create module
        module_class, module_kwargs, returns_weights = create_attention_module(operation, config)
        module = module_class(**module_kwargs)

        # Initialize parameters with dummy input
        rng_key, init_key = random.split(rng_key)
        params = module.init(init_key, x, mask=mask, training=False)

        # Execute forward pass
        if returns_weights:
            output, attention_weights = module.apply(params, x, mask=mask, training=False)
            result = {
                'success': True,
                'output': tensor_to_json(output),
                'attention_weights': tensor_to_json(attention_weights)
            }
        else:
            output = module.apply(params, x, mask=mask, training=False)
            result = {
                'success': True,
                'output': tensor_to_json(output)
            }

        return result

    except Exception as e:
        return {
            'success': False,
            'error': f"{type(e).__name__}: {str(e)}"
        }


def main():
    """
    Main REPL loop.

    Sends ready signal, then waits for JSON requests on stdin.
    For each request, executes attention operation and returns JSON response.
    """
    # Send ready signal
    print(json.dumps({"status": "ready"}), flush=True)

    # Initialize JAX random key
    rng_key = random.PRNGKey(42)

    # REPL loop
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
                rng_key, request_key = random.split(rng_key)
                response = handle_request(request, request_key)
                print(json.dumps(response), flush=True)

            except json.JSONDecodeError as e:
                error_response = {
                    'success': False,
                    'error': f"JSON decode error: {str(e)}"
                }
                print(json.dumps(error_response), flush=True)

            except Exception as e:
                error_response = {
                    'success': False,
                    'error': f"Unexpected error: {str(e)}"
                }
                print(json.dumps(error_response), flush=True)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
