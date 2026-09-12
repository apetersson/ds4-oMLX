#!/usr/bin/env python3
"""Synthetic coordinate fixture only; does not discover refusal directions."""
import argparse
import struct
from pathlib import Path
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('output', type=Path)
p.add_argument('--site', choices=['writer', 'residual'], required=True)
p.add_argument('--model-sha256', required=True)
a = p.parse_args()
digest = bytes.fromhex(a.model_sha256)
if len(digest) != 32:
    p.error('model SHA256 must contain 64 hex digits')
header = struct.pack('<8s6IQ32s', b'DS41DIR\0', 1, 40, 5120, 1,
                     1 if a.site == 'writer' else 2, 0, (1 << 40) - 1, digest)
row = struct.pack('<5120f', 1.0, *([0.0] * 5119))
a.output.write_bytes(header + row * 40)
