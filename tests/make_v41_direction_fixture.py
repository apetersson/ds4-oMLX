"""Write an artificial first-coordinate writer direction for runtime tests."""
import argparse
import struct

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('output')
parser.add_argument('model_sha256')
parser.add_argument('--site', type=int, choices=[1, 2], default=1)
args = parser.parse_args()
digest = bytes.fromhex(args.model_sha256)
if len(digest) != 32:
    parser.error('model SHA256 must be exactly 32 bytes')
header = struct.pack('<8s6IQ32s', b'DS41DIR\0', 1, 40, 5120, 1, args.site, 0, 1, digest)
with open(args.output, 'wb') as stream:
    stream.write(header)
    stream.write(struct.pack('<f', 1.0))
    stream.write(bytes(40 * 5120 * 4 - 4))
