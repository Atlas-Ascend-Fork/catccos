"""Out-of-process CATCCOS lowering for ``catlass.execution``.

The Catlass Python bridge and ``TlaDistCompile`` both statically link LLVM/MLIR.
Calling the TLA_DIST pipeline through the Python capsule bridge can therefore put
objects owned by different MLIR images in the same process. Use the TLA_DIST
command-line compiler as the process boundary and exchange only plain MLIR text
with it.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from catlass.compiler_bridge import (
    KernelAbiArgument,
    KernelAbiArgumentKind,
    KernelAbiFloatFormat,
    KernelAbiIntegerSignedness,
    KernelAbiLayout,
    KernelAbiScalarCategory,
    KernelAbiScalarDescriptor,
    TlaLoweringResult,
)
from catlass.execution import TlaKernelCompileError, TlaRuntimeUnavailableError


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_tla_dist_compile() -> Path | None:
    """Locate the TLA_DIST command-line compiler without requiring an env script."""
    candidates: list[Path] = []
    explicit = os.getenv("TLA_DIST_DSL_COMPILE")
    if explicit:
        candidates.append(Path(explicit).expanduser())

    package_dir = Path(__file__).resolve().parent
    project_root = _project_root()
    candidates.extend(
        (
            package_dir / "bin" / "TlaDistCompile",
            project_root / "csrc" / "mlir" / "build" / "python" / "catccos" / "tla_dist" / "bin" / "TlaDistCompile",
            project_root / "csrc" / "mlir" / "build" / "tools" / "tla_dist-compile" / "TlaDistCompile",
        )
    )
    which = shutil.which("TlaDistCompile")
    if which:
        candidates.append(Path(which))

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    return None


def build_kernel_abi(lowered_module: Any) -> KernelAbiLayout:
    """Read the entry signature from the existing frontend MLIR module."""
    from catlass._mlir import ir as mlir_ir

    with lowered_module.context:
        for view in lowered_module.body.operations:
            operation = view.operation
            if operation.name != "tla.func":
                continue
            entrypoint = mlir_ir.StringAttr(operation.attributes["sym_name"]).value
            function_type = mlir_ir.FunctionType(mlir_ir.TypeAttr(operation.attributes["function_type"]).value)
            argument_types = [str(argument_type) for argument_type in function_type.inputs]
            return _build_kernel_abi_from_signature(entrypoint, argument_types)
    raise TlaKernelCompileError("TLA_DIST module has no tla.func entrypoint")


def _scalar_descriptor(mlir_type: str) -> tuple[KernelAbiScalarDescriptor, int] | None:
    token = mlir_type.strip()
    if token == "index":
        return (
            KernelAbiScalarDescriptor(
                category=KernelAbiScalarCategory.INDEX,
                bit_width=64,
                integer_signedness=None,
                float_format=None,
            ),
            8,
        )
    integer = re.fullmatch(r"([su]?)i(1|8|16|32|64)", token)
    if integer:
        prefix, width_text = integer.groups()
        width = int(width_text)
        signedness = {
            "s": KernelAbiIntegerSignedness.SIGNED,
            "u": KernelAbiIntegerSignedness.UNSIGNED,
            "": KernelAbiIntegerSignedness.SIGNLESS,
        }[prefix]
        return (
            KernelAbiScalarDescriptor(
                category=KernelAbiScalarCategory.INTEGER,
                bit_width=width,
                integer_signedness=signedness,
                float_format=None,
            ),
            max(1, width // 8),
        )
    float_formats = {
        "f16": (KernelAbiFloatFormat.F16, 16),
        "bf16": (KernelAbiFloatFormat.BF16, 16),
        "f32": (KernelAbiFloatFormat.F32, 32),
    }
    float_info = float_formats.get(token)
    if float_info is None:
        return None
    float_format, width = float_info
    return (
        KernelAbiScalarDescriptor(
            category=KernelAbiScalarCategory.FLOAT,
            bit_width=width,
            integer_signedness=None,
            float_format=float_format,
        ),
        width // 8,
    )


def _is_gm_pointer_type(mlir_type: str) -> bool:
    compact = re.sub(r"\s+", "", mlir_type)
    return compact.startswith(("!tla.ptr<", "!tla.tensor<")) and re.search(r",gm(?:,|>)", compact) is not None


def _gm_tensor_descriptor_info(mlir_type: str) -> tuple[tuple[str, ...], bool]:
    """Describe a rank-1/2 GM tensor after Catlass bridges it to a memref."""
    compact = re.sub(r"\s+", "", mlir_type)
    layout = re.search(
        r"!tla\.layout<!tla\.shape<([^<>]+)>,!tla\.stride<([^<>]+)>,"
        r"!tla\.shape<[^<>]+>,(RowMajor|ColumnMajor)>",
        compact,
    )
    if layout is None:
        raise TlaKernelCompileError(f"unsupported TLA_DIST GM tensor layout: {mlir_type!r}")
    shape = re.findall(r"\?|\d+", layout.group(1))
    strides = re.findall(r"\?|\d+", layout.group(2))
    rank = len(shape)
    if rank not in (1, 2) or len(strides) != rank:
        raise TlaKernelCompileError(f"unsupported TLA_DIST GM tensor rank: {mlir_type!r}")

    requires_descriptor = "?" in shape or "?" in strides
    if layout.group(3) == "RowMajor" and not requires_descriptor:
        # Catlass gives noncompact row-major tensors a dynamic memref offset,
        # so even a fully static padded tensor selects descriptor mode.
        expected_stride = 1
        for extent, stride in zip(reversed(shape), reversed(strides)):
            requires_descriptor |= int(stride) != expected_stride
            expected_stride *= int(extent)
    fields = (
        "allocated",
        "aligned",
        "offset",
        *(f"size{dim}" for dim in range(rank)),
        *(f"stride{dim}" for dim in range(rank)),
    )
    return fields, requires_descriptor


def _is_dynamic_gm_memref_type(mlir_type: str) -> bool:
    # The frontend dynamic-GM ABI always uses four dynamic size/stride slots,
    # even when some logical layout fields are compile-time constants.
    compact = re.sub(r"\s+", "", mlir_type)
    return (
        re.fullmatch(
            r"memref<\?x\?x\?x\?x[^,< >]+,"
            r"strided<\[\?,\?,\?,\?\],offset:\?>,"
            r"#hivm\.address_space<gm>>",
            compact,
        )
        is not None
    )


def _build_kernel_abi_from_signature(entrypoint: str, argument_types: list[str]) -> KernelAbiLayout:
    """Build the launch ABI described by the pinned Catlass lowering."""
    arguments: list[KernelAbiArgument] = []
    offset = 0
    any_memref = False
    memref_fields = (
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

    def append(
        *,
        logical_index: int,
        kind: KernelAbiArgumentKind,
        mlir_type: str,
        storage_size: int,
        scalar: KernelAbiScalarDescriptor | None = None,
        field: str | None = None,
    ) -> None:
        nonlocal offset
        offset = (offset + 3) & ~3
        arguments.append(
            KernelAbiArgument(
                index=len(arguments),
                logical_index=logical_index,
                kind=kind,
                scalar=scalar,
                mlir_type=mlir_type,
                offset=offset,
                storage_size=storage_size,
                alignment=4,
                field=field,
            )
        )
        offset += storage_size

    # A dynamic host tensor occupies three IR arguments. Consume its companion
    # origins here so all 13 launch fields retain the same host logical index.
    logical_types: list[tuple[str, bool]] = []
    ir_index = 0
    while ir_index < len(argument_types):
        mlir_type = argument_types[ir_index]
        dynamic_gm = _is_dynamic_gm_memref_type(mlir_type)
        if dynamic_gm:
            if argument_types[ir_index + 1 : ir_index + 3] != ["index", "index"]:
                raise TlaKernelCompileError(
                    f"TLA_DIST dynamic GM memref argument {ir_index} requires two following index origin arguments"
                )
            ir_index += 3
        else:
            ir_index += 1
        logical_types.append((mlir_type, dynamic_gm))

    tensor_descriptors = {
        index: _gm_tensor_descriptor_info(mlir_type)
        for index, (mlir_type, _) in enumerate(logical_types)
        if mlir_type.lstrip().startswith("!tla.tensor<") and _is_gm_pointer_type(mlir_type)
    }
    # The device entry uses one memref calling convention. Once any GM memref
    # requires a descriptor, static GM tensors must also expand by their rank.
    requires_memref_descriptors = any(dynamic_gm for _, dynamic_gm in logical_types) or any(
        required for _, required in tensor_descriptors.values()
    )
    for logical_index, (mlir_type, dynamic_gm) in enumerate(logical_types):
        if dynamic_gm or _is_gm_pointer_type(mlir_type):
            fields = memref_fields if dynamic_gm else ()
            if requires_memref_descriptors and logical_index in tensor_descriptors:
                fields = tensor_descriptors[logical_index][0]
            if fields:
                any_memref = True
                for field in fields:
                    append(
                        logical_index=logical_index,
                        kind=KernelAbiArgumentKind.MEMREF_FIELD,
                        mlir_type=mlir_type,
                        storage_size=8,
                        field=field,
                    )
            else:
                append(
                    logical_index=logical_index,
                    kind=KernelAbiArgumentKind.POINTER,
                    mlir_type=mlir_type,
                    storage_size=8,
                )
            continue

        scalar_info = _scalar_descriptor(mlir_type)
        if scalar_info is None:
            raise TlaKernelCompileError(
                f"TLA_DIST CLI lowering cannot describe launch ABI argument {logical_index} with type {mlir_type!r}"
            )
        scalar, storage_size = scalar_info
        append(
            logical_index=logical_index,
            kind=KernelAbiArgumentKind.SCALAR,
            mlir_type=mlir_type,
            storage_size=storage_size,
            scalar=scalar,
        )

    return KernelAbiLayout(
        schema_version=4 if any_memref else 3,
        entrypoint=entrypoint,
        total_size=(offset + 7) & ~7,
        arguments=tuple(arguments),
    )


def lower_tlair_to_mlir_cli(
    *,
    tlair_mlir: str,
    kernel_abi: KernelAbiLayout,
    mlir_path: Path,
    compile_option: Any | None = None,
) -> TlaLoweringResult:
    """Lower TLAIR with ``TlaDistCompile`` and return compiler launch metadata."""
    compiler = resolve_tla_dist_compile()
    if compiler is None:
        raise TlaRuntimeUnavailableError(
            "TlaDistCompile was not found. Run python/tla_dist_dsl/build.sh or set "
            "TLA_DIST_DSL_COMPILE to the TlaDistCompile executable."
        )

    input_path = mlir_path.with_suffix(".tlair.mlir")
    input_path.write_text(tlair_mlir, encoding="utf-8")
    command = [
        str(compiler),
        str(input_path),
        "--emit=mlir",
        "-o",
        str(mlir_path),
    ]
    print_requested = bool(compile_option is not None and compile_option.print_ir)
    if print_requested:
        command.append("--mlir-print-ir-before-all")
        command.append("--mlir-print-ir-after-all")

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        raise TlaKernelCompileError(
            f"TlaDistCompile failed with exit code {exc.returncode}\n"
            f"cmd: {' '.join(command)}\n"
            f"stdout:\n{exc.stdout or ''}\n"
            f"stderr:\n{'<captured in pass IR dump>' if print_requested else stderr}",
            pass_ir_dump=stderr if print_requested else "",
        ) from exc

    if not mlir_path.is_file():
        raise TlaKernelCompileError(f"TlaDistCompile completed without producing lowered MLIR at {mlir_path}")
    lowered_mlir = mlir_path.read_text(encoding="utf-8")

    def module_integer(name: str, default: int = 0) -> int:
        # Preserve launch metadata across the textual compiler boundary.
        match = re.search(rf"\btla\.{name}\s*=\s*(-?\d+)\s*:\s*i64\b", lowered_mlir)
        return int(match.group(1)) if match is not None else default

    return TlaLoweringResult(
        lowered_mlir=lowered_mlir,
        pass_ir_dump=completed.stderr if print_requested else "",
        kernel_abi=kernel_abi,
        ub_static_bytes=module_integer("ub_static_bytes"),
        ub_programmable_bytes=module_integer("ub_programmable_bytes"),
        ub_dynamic_base=module_integer("ub_dynamic_base", -1),
    )


def bridge_path() -> Path | None:
    """Return the compiler executable used as the TLA_DIST cache fingerprint."""
    return resolve_tla_dist_compile()


__all__ = [
    "bridge_path",
    "build_kernel_abi",
    "lower_tlair_to_mlir_cli",
    "resolve_tla_dist_compile",
]
