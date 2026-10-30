"""Dump real example IR without initializing ACL, SHMEM or a process group."""

from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[4]
example = runpy.run_path(str(root / 'examples' / sys.argv[1]))
print(example['dump_tlair']())
