"""Check template selection without importing native Catlass extensions."""

from __future__ import annotations

import ast
import hashlib
import os
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Runtime:
    kernel_mode: str
    arch_scope: str


class BitcodeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / "catccos" / "tla_dist"
        source = ROOT / "catccos" / "tla_dist" / "compiler.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        functions = ast.Module(
            body=[
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name
                in {
                    "_project_root",
                    "_resolve_base_bitcode",
                    "_bitcode_fingerprint",
                    "_catccos_execution_hooks",
                }
            ],
            type_ignores=[],
        )
        self.scope = {
            "__file__": str(self.package / "compiler.py"),
            "Path": Path,
            "os": os,
            "Any": object,
            "hashlib": hashlib,
            "TlaRuntimeUnavailableError": RuntimeError,
            "contextmanager": contextmanager,
            "replace": replace,
            "Iterator": Iterator,
            "bridge_path": lambda: None,
        }
        exec(compile(functions, str(source), "exec"), self.scope)
        self.env = patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop("TLA_DIST_DSL_BASE_BITCODE", None)

    def write(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test bitcode")
        return str(path.resolve())

    def resolve(self, mode):
        return self.scope["_resolve_base_bitcode"](
            Runtime(kernel_mode=mode, arch_scope="aiv.c310" if mode == "aiv" else "aic.c310")
        )

    def test_packaged_templates_are_separated_by_mode(self):
        bc = self.package / "bc"
        aic = self.write(bc / "meta_op.aic.c310.bc")
        aiv = self.write(bc / "meta_op.aiv.c310.bc")
        mixed = self.write(bc / "meta_op.mix_aiv.c310.bc")
        self.assertEqual(self.resolve("aiv"), aiv)
        self.assertEqual(self.resolve("aic"), aic)
        self.assertEqual(self.resolve("mix"), f"{aic},{mixed}")

    def test_mix_does_not_fall_back_to_soft_barrier(self):
        bc = self.package / "bc"
        aic = self.write(bc / "meta_op.aic.c310.bc")
        aiv = self.write(bc / "meta_op.aiv.c310.bc")
        self.assertEqual(self.resolve("aic"), aic)
        self.assertEqual(self.resolve("aiv"), aiv)
        with self.assertRaisesRegex(RuntimeError, "meta_op.mix_aiv.c310.bc"):
            self.assertIsNotNone(self.resolve("mix"))

    def test_raw_build_fallback_uses_mixed_template(self):
        build = self.root / "csrc" / "mlir" / "build"
        aic = self.write(build / "tla" / "bc" / "meta_op.aic.c310.bc")
        aiv = self.write(build / "bc" / "meta_op.tla_dist.aiv.c310.bc")
        mixed = self.write(build / "bc" / "meta_op.tla_dist.mix_aiv.c310.bc")
        self.assertEqual(self.resolve("mix"), f"{aic},{mixed}")
        self.assertEqual(self.resolve("aiv"), aiv)

    def test_explicit_bitcode_override_is_preserved(self):
        explicit = self.write(self.root / "custom.bc")
        os.environ["TLA_DIST_DSL_BASE_BITCODE"] = explicit
        self.assertEqual(self.resolve("mix"), explicit)

    def test_initial_aic_cache_tracks_mixed_bitcode_changes(self):
        bc = self.package / "bc"
        aic = self.write(bc / "meta_op.aic.c310.bc")
        aiv = self.write(bc / "meta_op.aiv.c310.bc")
        self.assertEqual(self.resolve("aic"), aic)
        self.assertEqual(self.resolve("aiv"), aiv)
        mixed = Path(self.write(bc / "meta_op.mix_aiv.c310.bc"))
        execution = SimpleNamespace(
            _run_tla_lowering_to_mlir=None,
            resolve_bridge_extension_path=None,
            _resolve_hivm_template_bitcode=None,
            _cache_key=lambda **kwargs: kwargs["tlair_mlir"],
        )
        self.scope["_execution"] = execution
        with self.scope["_catccos_execution_hooks"]():

            def key(mode):
                return execution._cache_key(compile_option=Runtime(mode, f"{mode}.c310"), tlair_mlir="kernel")

            initial_aic = key("aic")
            initial_aiv = key("aiv")
            mixed.write_bytes(b"updated hard synchronization bitcode")
            self.assertNotEqual(key("aic"), initial_aic)
            self.assertEqual(key("aiv"), initial_aiv)

    def test_execution_hook_accepts_catlass_variant_keyword(self):
        bc = self.package / "bc"
        aic = self.write(bc / "meta_op.aic.c310.bc")
        mixed = self.write(bc / "meta_op.mix_aiv.c310.bc")
        original = object()
        execution = SimpleNamespace(
            _run_tla_lowering_to_mlir=None,
            resolve_bridge_extension_path=None,
            _resolve_hivm_template_bitcode=original,
            _cache_key=None,
        )
        self.scope["_execution"] = execution
        with self.scope["_catccos_execution_hooks"]():
            runtime = Runtime("mix", "aic.c310")
            self.assertEqual(
                execution._resolve_hivm_template_bitcode(runtime, variant="normal"),
                f"{aic},{mixed}",
            )
            with self.assertRaisesRegex(RuntimeError, "print.*is not built"):
                execution._resolve_hivm_template_bitcode(runtime, variant="print")
        self.assertIs(execution._resolve_hivm_template_bitcode, original)

    def test_execution_hook_preserves_explicit_print_bitcode_override(self):
        explicit = self.write(self.root / "custom-print-with-shmem.bc")
        os.environ["TLA_DIST_DSL_BASE_BITCODE"] = explicit
        execution = SimpleNamespace(
            _run_tla_lowering_to_mlir=None,
            resolve_bridge_extension_path=None,
            _resolve_hivm_template_bitcode=None,
            _cache_key=None,
        )
        self.scope["_execution"] = execution
        with self.scope["_catccos_execution_hooks"]():
            self.assertEqual(
                execution._resolve_hivm_template_bitcode(Runtime("mix", "aic.c310"), variant="print"),
                explicit,
            )


if __name__ == "__main__":
    unittest.main()
