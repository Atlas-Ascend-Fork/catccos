"""Build compute templates using the pinned Catlass bitcode compiler."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catlass-dsl-root", type=Path, required=True)
    parser.add_argument("--ascend-home", type=Path, required=True)
    parser.add_argument("--compiler-bin", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "_catccos_bc_compile", args.catlass_dsl_root / "catlass/bc_compile.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Catlass bitcode compiler from {args.catlass_dsl_root}")
    bc_compile = importlib.util.module_from_spec(spec)
    # dataclass requires its module to be registered while classes are defined.
    sys.modules[spec.name] = bc_compile
    spec.loader.exec_module(bc_compile)

    bc_root = args.catlass_dsl_root / "csrc/mlir/bc"
    config = bc_compile.BCConfig(
        paths=bc_compile.BCPaths(
            bc_stubs=bc_root,
            catlass_include=args.catlass_dsl_root.parents[1] / "include",
            bc_include=bc_root,
        ),
        toolchain=bc_compile.BCToolchain(
            ccec=args.compiler_bin / "ccec",
            llvm_link=args.compiler_bin / "llvm-link",
            ascend_home=args.ascend_home,
        ),
        cce_arch="c310",
        catlass_arch=3510,
        cache_dir=args.output_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # CMake invokes this only when sources, headers or the toolchain change.
    # Invalidate this build's templates so build_meta_op also rebuilds stubs.
    for artifact in args.output_dir.glob("*.bc"):
        artifact.unlink()
    bc_compile.build_meta_op(["aic", "aiv"], args.output_dir, config, variant="normal")


if __name__ == "__main__":
    main()
