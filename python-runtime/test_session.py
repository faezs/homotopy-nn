#!/usr/bin/env python3
"""
Test script for einsum_session.py

Simulates Haskell bridge by sending JSON requests and checking responses.
"""

import json
import subprocess
import sys


def test_matmul():
    """
    Test matrix multiplication: A[j,i] × B[j,k] → C[i,k]

    Note: Contract constructor puts contracted indices FIRST.
    So (contract [j] [[i] [k]]) means:
    - Input 1: [j, i] (contracted ++ remaining[0])
    - Input 2: [j, k] (contracted ++ remaining[1])
    - Output: [i, k] (remaining[0] ++ remaining[1])

    Formula: "ji,jk->ik"
    """
    print("Test 1: Matrix multiplication")
    print("=" * 60)

    request = {
        'formula': '(contract [j] [[i] [k]])',
        'tensors': [
            {
                # Shape [j, i] = [3, 2] (contracted index first!)
                'shape': [3, 2],
                'data': [1.0, 2.0,   # j=0
                        3.0, 4.0,   # j=1
                        5.0, 6.0]   # j=2
            },
            {
                # Shape [j, k] = [3, 2]
                'shape': [3, 2],
                'data': [7.0, 8.0,   # j=0
                        9.0, 10.0,  # j=1
                        11.0, 12.0] # j=2
            }
        ]
    }

    print(f"Request: {json.dumps(request, indent=2)}")

    # Send to session
    proc = subprocess.Popen(
        ['python3', 'einsum_session.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for ready signal
    ready = proc.stdout.readline()
    print(f"Session ready: {ready.strip()}")

    # Send request
    proc.stdin.write(json.dumps(request) + '\n')
    proc.stdin.flush()

    # Read response
    response_line = proc.stdout.readline()
    response = json.loads(response_line)

    print(f"Response: {json.dumps(response, indent=2)}")

    if response['success']:
        print(f"✅ Success! Result shape: {response['shape']}")
        print(f"   Result data: {response['data']}")
    else:
        print(f"❌ Error: {response['error']}")

    proc.stdin.close()
    proc.wait()
    print()


def test_dot_product():
    """Test dot product: v[i] · w[i] → scalar[]"""
    print("Test 2: Dot product")
    print("=" * 60)

    request = {
        'formula': '(contract [i] [[] []])',
        'tensors': [
            {
                'shape': [5],
                'data': [1.0, 2.0, 3.0, 4.0, 5.0]
            },
            {
                'shape': [5],
                'data': [1.0, 1.0, 1.0, 1.0, 1.0]
            }
        ]
    }

    print(f"Request: {json.dumps(request, indent=2)}")

    proc = subprocess.Popen(
        ['python3', 'einsum_session.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    ready = proc.stdout.readline()
    print(f"Session ready: {ready.strip()}")

    proc.stdin.write(json.dumps(request) + '\n')
    proc.stdin.flush()

    response_line = proc.stdout.readline()
    response = json.loads(response_line)

    print(f"Response: {json.dumps(response, indent=2)}")

    if response['success']:
        expected = 1.0 + 2.0 + 3.0 + 4.0 + 5.0  # 15.0
        result = response['data'][0]
        print(f"✅ Success! Result: {result} (expected: {expected})")
    else:
        print(f"❌ Error: {response['error']}")

    proc.stdin.close()
    proc.wait()
    print()


def test_transpose():
    """Test transpose: M[i,j] → Mᵀ[j,i]"""
    print("Test 3: Transpose")
    print("=" * 60)

    request = {
        'formula': '(transpose [i j] [j i])',
        'tensors': [
            {
                'shape': [2, 3],
                'data': [1.0, 2.0, 3.0,
                        4.0, 5.0, 6.0]
            }
        ]
    }

    print(f"Request: {json.dumps(request, indent=2)}")

    proc = subprocess.Popen(
        ['python3', 'einsum_session.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    ready = proc.stdout.readline()
    print(f"Session ready: {ready.strip()}")

    proc.stdin.write(json.dumps(request) + '\n')
    proc.stdin.flush()

    response_line = proc.stdout.readline()
    response = json.loads(response_line)

    print(f"Response: {json.dumps(response, indent=2)}")

    if response['success']:
        print(f"✅ Success! Result shape: {response['shape']}")
        print(f"   Original: [2, 3]")
        print(f"   Transposed: {response['shape']}")
        assert response['shape'] == [3, 2], "Shape mismatch!"
    else:
        print(f"❌ Error: {response['error']}")

    proc.stdin.close()
    proc.wait()
    print()


def test_reduce():
    """Test reduce: M[i,j] → v[i] (sum over j)"""
    print("Test 4: Reduce (sum over dimension)")
    print("=" * 60)

    request = {
        'formula': '(reduce [i j] j)',
        'tensors': [
            {
                'shape': [2, 3],
                'data': [1.0, 2.0, 3.0,  # Row 1: sum = 6.0
                        4.0, 5.0, 6.0]   # Row 2: sum = 15.0
            }
        ]
    }

    print(f"Request: {json.dumps(request, indent=2)}")

    proc = subprocess.Popen(
        ['python3', 'einsum_session.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    ready = proc.stdout.readline()
    print(f"Session ready: {ready.strip()}")

    proc.stdin.write(json.dumps(request) + '\n')
    proc.stdin.flush()

    response_line = proc.stdout.readline()
    response = json.loads(response_line)

    print(f"Response: {json.dumps(response, indent=2)}")

    if response['success']:
        print(f"✅ Success! Result shape: {response['shape']}")
        print(f"   Row sums: {response['data']}")
        print(f"   Expected: [6.0, 15.0]")
    else:
        print(f"❌ Error: {response['error']}")

    proc.stdin.close()
    proc.wait()
    print()


def test_error_handling():
    """Test error handling with malformed formula"""
    print("Test 5: Error handling")
    print("=" * 60)

    request = {
        'formula': '(invalid formula',  # Missing closing paren
        'tensors': []
    }

    print(f"Request: {json.dumps(request, indent=2)}")

    proc = subprocess.Popen(
        ['python3', 'einsum_session.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    ready = proc.stdout.readline()
    print(f"Session ready: {ready.strip()}")

    proc.stdin.write(json.dumps(request) + '\n')
    proc.stdin.flush()

    response_line = proc.stdout.readline()
    response = json.loads(response_line)

    print(f"Response: {json.dumps(response, indent=2)}")

    if not response['success']:
        print(f"✅ Error correctly reported: {response['error']}")
    else:
        print(f"❌ Should have failed but succeeded!")

    proc.stdin.close()
    proc.wait()
    print()


if __name__ == '__main__':
    print("Testing PyTorch Einsum Session")
    print("=" * 60)
    print()

    test_matmul()
    test_dot_product()
    test_transpose()
    test_reduce()
    test_error_handling()

    print("=" * 60)
    print("All tests completed!")
