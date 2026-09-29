#!/usr/bin/env python3
"""Export an already verified layout into Orbax, without reading source again."""

import argparse
from pathlib import Path

from convert_checkpoint import export

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    args = p.parse_args()
    export(args.checkpoint.resolve())
