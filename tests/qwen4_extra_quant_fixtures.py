#!/usr/bin/env python3
"""Generate deterministic GGML blocks and independent gguf-py reference values.
Requires numpy and gguf. No model download is needed.
"""
import argparse
from pathlib import Path
import numpy as np
from gguf import dequantize, GGMLQuantizationType

def main():
 parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 rng=np.random.default_rng(7349)
 for typ,block,bytes_,scales in [(7,32,24,[0,2]),(13,256,176,[0,2]),(14,256,210,[208]),(20,32,18,[0])]:
  raw=rng.integers(0,256,(262144//block,bytes_),dtype=np.uint8)
  for off in scales:
   h=rng.uniform(-0.125,0.125,raw.shape[0]).astype('<f2').view(np.uint8).reshape(-1,2);raw[:,off:off+2]=h
  ref=dequantize(raw.reshape(-1),GGMLQuantizationType(typ));assert np.isfinite(ref).all()
  (args.output/f'{typ}.bin').write_bytes(raw.tobytes());(args.output/f'{typ}.f32').write_bytes(ref.astype('<f4').tobytes())
if __name__=='__main__':main()
