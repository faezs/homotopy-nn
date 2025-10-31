#!/usr/bin/env python3
"""
Persistent PyTorch Einsum REPL Session

Reads JSON requests from stdin, executes einsum operations, returns results.

Protocol:
    Input (JSON line):
        {
            "formula": "(contract [j] [[i] [k]])",
            "tensors": [
                {"shape": [2, 3], "data": [1.0, 2.0, ...]},
                {"shape": [3, 4], "data": [...]}
            ]
        }

    Output (JSON line):
        {
            "success": true,
            "shape": [2, 4],
            "data": [19.0, 22.0, ...]
        }

    Error (JSON line):
        {
            "success": false,
            "error": "ParseError: Expected '(', got ..."
        }

Usage:
    # Start session (waits for input)
    $ python3 einsum_session.py

    # Send request
    $ echo '{"formula": "(contract [j] [[i] [k]])", "tensors": [...]}' | python3 einsum_session.py

    # From Haskell:
    import System.Process
    (Just stdin, Just stdout, _, _) = createProcess (proc "python3" ["einsum_session.py"])
                                                    { std_in = CreatePipe, std_out = CreatePipe }
    hPutStrLn stdin requestJSON
    responseJSON <- hGetLine stdout
"""

import sys
import json
import torch
from einsum_parser import EinsumParser, ParseError


def parse_tensor(tensor_json: dict) -> torch.Tensor:
    """
    Convert JSON tensor representation to PyTorch tensor.

    Args:
        tensor_json: {"shape": [2, 3], "data": [1.0, 2.0, ...]}

    Returns:
        PyTorch tensor with specified shape and data
    """
    shape = tensor_json['shape']
    data = tensor_json['data']
    return torch.tensor(data, dtype=torch.float32).reshape(shape)


def tensor_to_json(tensor: torch.Tensor) -> dict:
    """
    Convert PyTorch tensor to JSON representation.

    Args:
        tensor: PyTorch tensor

    Returns:
        {"shape": [...], "data": [...]}
    """
    return {
        'shape': list(tensor.shape),
        'data': tensor.flatten().tolist()
    }


def handle_request(request: dict, parser: EinsumParser) -> dict:
    """
    Process a single einsum request.

    Args:
        request: {"formula": "...", "tensors": [...]}
        parser: EinsumParser instance

    Returns:
        {"success": true, "shape": [...], "data": [...]}
        or {"success": false, "error": "..."}
    """
    try:
        formula = request['formula']
        tensor_jsons = request['tensors']

        # Parse tensors
        tensors = [parse_tensor(tj) for tj in tensor_jsons]

        # Parse and execute einsum formula
        executor = parser.parse(formula)
        result = executor(tensors)

        # Return result
        return {
            'success': True,
            **tensor_to_json(result)
        }

    except ParseError as e:
        return {
            'success': False,
            'error': f'ParseError: {str(e)}'
        }
    except ValueError as e:
        return {
            'success': False,
            'error': f'ValueError: {str(e)}'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'{type(e).__name__}: {str(e)}'
        }


def main():
    """Main REPL loop - read JSON from stdin, write JSON to stdout."""
    parser = EinsumParser()

    # Send ready signal
    print(json.dumps({'status': 'ready'}), flush=True)

    # Process requests line-by-line
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            response = handle_request(request, parser)
            print(json.dumps(response), flush=True)

        except json.JSONDecodeError as e:
            error_response = {
                'success': False,
                'error': f'JSONDecodeError: {str(e)}'
            }
            print(json.dumps(error_response), flush=True)


if __name__ == '__main__':
    main()
