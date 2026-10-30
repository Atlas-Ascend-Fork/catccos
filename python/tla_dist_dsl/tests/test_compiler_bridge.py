"""Launch ABI regressions runnable without MLIR, pytest, or an NPU."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MEMREF = "memref<?x?x?x?xf16, strided<[?, ?, ?, ?], offset: ?>, #hivm.address_space<gm>>"
FIELDS = (
    "allocated",
    "aligned",
    "offset",
    "size0",
    "size1",
    "size2",
    "size3",
    "stride0",
    "stride1",
    "stride2",
    "stride3",
    "originShape0",
    "originShape1",
)


def static_tensor_type(rank, padded=False):
    shape = "4" if rank == 1 else "4, 4"
    stride = ("2" if padded else "1") if rank == 1 else ("8, 1" if padded else "4, 1")
    coord = "0" if rank == 1 else "0, 0"
    return (
        f"!tla.tensor<!tla.layout<!tla.shape<{shape}>, !tla.stride<{stride}>, "
        f"!tla.shape<{shape}>, RowMajor>, !tla.coord<{coord}>, !tla.ptr<f16, gm, 4>>"
    )


def load_bridge():
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    # Use the real pinned ABI dataclasses, stubbing only native/package imports.
    catlass_root = ROOT.parents[1] / "3rdparty/catlass/python/tla_dsl/catlass"
    package = ModuleType("catlass")
    package.__path__ = [str(catlass_root)]
    execution = ModuleType("catlass.execution")
    execution.TlaKernelCompileError = type("TlaKernelCompileError", (RuntimeError,), {})
    execution.TlaRuntimeUnavailableError = RuntimeError
    with patch.dict(
        sys.modules,
        {
            "catlass": package,
            "catlass._tla_type_bridge": ModuleType("catlass._tla_type_bridge"),
            "catlass.execution": execution,
        },
    ):
        package.compiler_bridge = load(
            "catlass.compiler_bridge",
            catlass_root / "compiler_bridge.py",
        )
        return load("_test_dist_bridge", ROOT / "catccos/tla_dist/compiler_bridge.py")


class LaunchAbiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge = load_bridge()

    def abi(self, types):
        return self.bridge._build_kernel_abi_from_signature("kernel", types)

    def test_four_dynamic_tensors_and_rank_scalars(self):
        types = [MEMREF, "index", "index"] * 2
        types += [MEMREF.replace("f16", "f32"), "index", "index"] * 2
        types += ["i32", "i32"]
        abi = self.abi(types)
        self.assertEqual(abi.schema_version, 4)
        self.assertEqual(abi.entrypoint, "kernel")
        self.assertEqual(len(abi.arguments), 54)
        self.assertEqual(abi.total_size, 424)
        for logical in range(4):
            slots = abi.arguments[logical * 13 : (logical + 1) * 13]
            self.assertEqual(tuple(s.field for s in slots), FIELDS)
            self.assertTrue(all(s.logical_index == logical for s in slots))
            self.assertTrue(all(s.kind.value == "memref_field" for s in slots))
            self.assertTrue(all(s.storage_size == 8 for s in slots))
            self.assertEqual([s.offset for s in slots], list(range(logical * 104, (logical + 1) * 104, 8)))
        self.assertEqual([s.logical_index for s in abi.arguments[-2:]], [4, 5])
        self.assertEqual([s.offset for s in abi.arguments[-2:]], [416, 420])

    def test_mixed_arguments_preserve_logical_indices(self):
        types = ["i32", "!tla.ptr<f32, gm, 16>", MEMREF, "index", "index", "index"]
        abi = self.abi(types)
        self.assertEqual([s.logical_index for s in abi.arguments], [0, 1] + [2] * 13 + [3])
        self.assertEqual(abi.arguments[-1].kind.value, "scalar")
        self.assertEqual(abi.arguments[-1].scalar.category.value, "index")

    def test_static_arguments_keep_schema_three(self):
        abi = self.abi(["!tla.ptr<f32, gm, 16>", "i32"])
        self.assertEqual(abi.schema_version, 3)
        self.assertEqual([s.kind.value for s in abi.arguments], ["pointer", "scalar"])
        self.assertEqual([s.offset for s in abi.arguments], [0, 8])
        self.assertEqual(abi.total_size, 16)

    def test_static_tensor_without_dynamic_gm_keeps_pointer_abi(self):
        for rank in (1, 2):
            with self.subTest(rank=rank):
                abi = self.abi([static_tensor_type(rank), "i32"])
                self.assertEqual(abi.schema_version, 3)
                self.assertEqual([slot.kind.value for slot in abi.arguments], ["pointer", "scalar"])
                self.assertEqual([slot.offset for slot in abi.arguments], [0, 8])
                self.assertEqual(abi.total_size, 16)

    def test_static_and_dynamic_tensors_share_descriptor_mode(self):
        for rank in (1, 2):
            static_fields = ("allocated", "aligned", "offset") + tuple(f"size{i}" for i in range(rank))
            static_fields += tuple(f"stride{i}" for i in range(rank))
            for static_first in (False, True):
                with self.subTest(rank=rank, static_first=static_first):
                    static = [static_tensor_type(rank)]
                    dynamic = [MEMREF, "index", "index"]
                    types = (static + dynamic if static_first else dynamic + static) + ["i32"]
                    abi = self.abi(types)
                    field_groups = (static_fields, FIELDS) if static_first else (FIELDS, static_fields)
                    expected = []
                    for logical, fields in enumerate(field_groups):
                        expected.extend((logical, field) for field in fields)
                    self.assertEqual(abi.schema_version, 4)
                    self.assertEqual([(slot.logical_index, slot.field) for slot in abi.arguments[:-1]], expected)
                    self.assertTrue(all(slot.kind.value == "memref_field" for slot in abi.arguments[:-1]))
                    self.assertTrue(all(slot.storage_size == 8 for slot in abi.arguments[:-1]))
                    descriptor_bytes = 8 * (len(static_fields) + 13)
                    self.assertEqual([slot.offset for slot in abi.arguments[:-1]], list(range(0, descriptor_bytes, 8)))
                    self.assertEqual([slot.index for slot in abi.arguments], list(range(len(expected) + 1)))
                    self.assertEqual(abi.arguments[-1].logical_index, 2)
                    self.assertEqual(abi.arguments[-1].kind.value, "scalar")
                    self.assertEqual(abi.arguments[-1].offset, descriptor_bytes)
                    self.assertEqual(abi.total_size, descriptor_bytes + 8)

    def test_padded_static_tensor_selects_descriptor_mode_for_entire_entry(self):
        types = [static_tensor_type(2), static_tensor_type(2, padded=True), "!tla.ptr<f32, gm, 16>", "i32"]
        abi = self.abi(types)
        fields = ("allocated", "aligned", "offset", "size0", "size1", "stride0", "stride1")
        self.assertEqual(abi.schema_version, 4)
        self.assertEqual(
            [(slot.logical_index, slot.field) for slot in abi.arguments[:14]],
            [(logical, field) for logical in (0, 1) for field in fields],
        )
        self.assertEqual([slot.kind.value for slot in abi.arguments[-2:]], ["pointer", "scalar"])
        self.assertEqual([slot.logical_index for slot in abi.arguments[-2:]], [2, 3])
        self.assertEqual([slot.offset for slot in abi.arguments[-2:]], [112, 120])
        self.assertEqual(abi.total_size, 128)

    def test_column_major_static_tensor_uses_entry_descriptor_mode(self):
        tensor = static_tensor_type(2).replace("4, 1", "1, 4").replace("RowMajor", "ColumnMajor")
        static_abi = self.abi([tensor])
        self.assertEqual(static_abi.schema_version, 3)
        self.assertEqual(static_abi.arguments[0].kind.value, "pointer")
        mixed_abi = self.abi([tensor, MEMREF, "index", "index"])
        self.assertEqual(len(mixed_abi.arguments), 20)
        self.assertEqual(mixed_abi.total_size, 160)
        self.assertEqual([slot.logical_index for slot in mixed_abi.arguments], [0] * 7 + [1] * 13)

    def test_tensor_descriptor_without_origin_companions_uses_its_rank(self):
        for rank, field_count in ((1, 5), (2, 7)):
            tensor = static_tensor_type(rank)
            dynamic = tensor.replace("!tla.shape<4", "!tla.shape<?", 1)
            for descriptor_tensor in (dynamic, static_tensor_type(rank, padded=True)):
                with self.subTest(rank=rank, tensor=descriptor_tensor):
                    abi = self.abi([tensor, descriptor_tensor, "i32"])
                    self.assertEqual(abi.schema_version, 4)
                    self.assertEqual(len(abi.arguments), 2 * field_count + 1)
                    self.assertEqual(
                        [slot.logical_index for slot in abi.arguments], [0] * field_count + [1] * field_count + [2]
                    )
                    self.assertEqual(abi.arguments[-1].offset, 16 * field_count)

    def test_missing_or_invalid_origin_arguments_are_rejected(self):
        for suffix in ([], ["index"], ["index", "i32"], ["i64", "index"]):
            with self.subTest(suffix=suffix):
                self.assertRaisesRegex(
                    self.bridge.TlaKernelCompileError, "two following index", self.abi, [MEMREF] + suffix
                )

    def test_other_memref_abis_are_not_silently_accepted(self):
        for ty in (MEMREF.replace("?x?x?x?x", "?x?x"), MEMREF.replace("<gm>", "<ub>")):
            with self.subTest(ty=ty):
                self.assertRaisesRegex(
                    self.bridge.TlaKernelCompileError, "cannot describe", self.abi, [ty, "index", "index"]
                )

    def test_cli_lowering_preserves_ub_launch_metadata(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "lowered.mlir"
            kernel_abi = self.abi([MEMREF, "index", "index"])
            self.assertEqual(kernel_abi.schema_version, 4)
            module = (
                "module attributes {tla.ub_static_bytes = 65536 : i64, "
                "tla.ub_programmable_bytes = 253952 : i64, "
                "tla.ub_dynamic_base = -1 : i64} {}"
            )

            def run(command, **kwargs):
                output.write_text(module, encoding="utf-8")
                self.assertIn("--mlir-print-ir-before-all", command)
                return SimpleNamespace(stderr="pass dump")

            with (
                patch.object(self.bridge, "resolve_tla_dist_compile", return_value=Path("/compiler")),
                patch.object(self.bridge.subprocess, "run", side_effect=run),
            ):
                result = self.bridge.lower_tlair_to_mlir_cli(
                    tlair_mlir="module {}",
                    kernel_abi=kernel_abi,
                    mlir_path=output,
                    compile_option=SimpleNamespace(print_ir=True),
                )
            self.assertEqual(result.ub_static_bytes, 65536)
            self.assertEqual(result.ub_programmable_bytes, 253952)
            self.assertEqual(result.ub_dynamic_base, -1)
            self.assertEqual(result.pass_ir_dump, "pass dump")
            self.assertIs(result.kernel_abi, kernel_abi)


if __name__ == "__main__":
    unittest.main()
