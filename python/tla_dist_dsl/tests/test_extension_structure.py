from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_python_sources_parse() -> None:
    for source in list((ROOT / "catccos").rglob("*.py")) + list((ROOT / "examples").rglob("*.py")):
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source))


def test_example_preserves_torch_owned_acl_lifecycle() -> None:
    source = _text("examples/fusion_ops/matmul_reduce_scatter.py")
    cleanup = source[source.index("    finally:", source.index("def run(")) :]
    assert "\n        tla.initialize(" not in source
    assert cleanup.index("shmem_runtime.finalize()") < cleanup.index("dist.destroy_process_group()")


def test_device_symbols_are_catccos_owned() -> None:
    source = _text("csrc/mlir/bc/Vector/shmem.cpp")
    assert '#include "dma_common.h"' in source
    assert '#include "catccos/comm/tile/tile_remote_copy.hpp"' in source
    assert not (ROOT / "csrc/mlir/bc/device_common.h").exists()
    assert "_mlir_ciface_shmem_remote_##Name##_gm_row_major_to_gm_row_major_##DType" in source
    assert "shmemRemoteCopyGmRowMajorToGmRowMajor" in source
    assert "Catccos::Comm::Tile::TileRemoteCopy<" in source
    gm_to_gm = source[source.index("shmemRemoteCopyGmRowMajorToGmRowMajor") : source.index('extern "C"')]
    assert "while (" not in gm_to_gm
    assert re.search(r"memref_t<__ubuf__\s+T,\s*1>\s*\*\s*ub\b", gm_to_gm)
    assert len(re.findall(r"const\s+TensorDesc\s*&\s*ubDesc\b", gm_to_gm)) == 1
    assert "ubDesc.coord0" in gm_to_gm
    assert "ubDesc.coord1" in gm_to_gm
    assert "AscendC::LocalTensor<T> ubTensor" in gm_to_gm
    assert "ACLSHMEM_INTERNAL_UB_BUF_START_ADDR" not in gm_to_gm
    assert "DESC_ABI_PARAMS(ub)" in source
    assert "TENSOR_DESC_12(ub)" in source
    assert "shmem_barrier_all_vec" in source
    assert "shmem_set_sync_base_addr" not in source
    assert "catccos_shmem" not in source


def test_top_level_build_is_opt_in() -> None:
    top_level = (ROOT.parents[1] / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "option(CATCCOS_ENABLE_TLA_DIST_DSL" in top_level
    assert "if(CATCCOS_ENABLE_TLA_DIST_DSL)" in top_level


def test_mixed_shmem_helpers_preserve_plain_c_abi_codegen() -> None:
    source = _text("csrc/mlir/bc/CMakeLists.txt")
    mixed_options = source.split('if(core_type STREQUAL "mix_aiv")', 1)[1]
    mixed_options = mixed_options.split("elseif", 1)[0]
    options = "\n".join(line for line in mixed_options.splitlines() if not line.lstrip().startswith("#"))
    assert "-D__CCE_AICORE_ENABLE_MIX__=1" in options
    assert "-DTLA_DIST_REQUIRE_MIX_SYNC" in options
    assert "-cce-enable-mix" not in options
    assert "-enable-mix=true" not in options


def test_package_prefers_pinned_catlass_without_overriding_its_bridge() -> None:
    package = _text("catccos/__init__.py")
    assert not (ROOT / "set_env.sh").exists()
    assert not (ROOT / "run_communication_ops.sh").exists()
    assert "CATLASS_DSL_TYPE_BRIDGE_EXTENSION" not in package
    assert "TLA_DSL_TYPE_BRIDGE_EXTENSION" not in package
    assert "TLA_DIST_DSL_COMPILER_BRIDGE_EXTENSION" not in package
    assert "_CatlassBridgeFinder" not in package
    assert "_install_catlass_bridge_compat" not in package


@pytest.mark.parametrize("pinned,preloaded", [(True, False), (False, False), (True, True)])
def test_catlass_source_selection_preserves_other_imports(tmp_path, pinned, preloaded) -> None:
    source_root = tmp_path / "project/python/tla_dist_dsl"
    package = source_root / "catccos"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(_text("catccos/__init__.py"), encoding="utf-8")
    installed_root = tmp_path / "installed"
    installed_catlass = installed_root / "catlass"
    installed_catlass.mkdir(parents=True)
    (installed_catlass / "__init__.py").write_text('origin = "installed"\n', encoding="utf-8")
    (installed_root / "import_probe.py").write_text('origin = "installed"\n', encoding="utf-8")
    if pinned:
        pinned_root = tmp_path / "project/3rdparty/catlass/python/tla_dsl"
        pinned_catlass = pinned_root / "catlass"
        pinned_catlass.mkdir(parents=True)
        (pinned_catlass / "__init__.py").write_text('from . import sibling\norigin = "pinned"\n', encoding="utf-8")
        (pinned_catlass / "sibling.py").write_text('origin = "pinned"\n', encoding="utf-8")
        (pinned_root / "import_probe.py").write_text('origin = "pinned"\n', encoding="utf-8")
    expected = "pinned" if pinned and not preloaded else "installed"
    script = f"""
import sys
sys.path.extend([{str(source_root)!r}, {str(installed_root)!r}])
if {preloaded!r}:
    import catlass
original_path = sys.path.copy()
import catccos
assert sys.path == original_path
assert ('catlass' in sys.modules) == {preloaded!r}
import catlass
assert catlass.origin == {expected!r}
if catlass.origin == 'pinned':
    assert catlass.sibling.origin == 'pinned'
import import_probe
assert import_probe.origin == 'installed'
"""
    result = subprocess.run([sys.executable, "-I", "-S", "-c", script], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_catlass_native_bridge_and_tla_dist_compiler_are_separate() -> None:
    cmake = _text("csrc/mlir/lib/Tools/CMakeLists.txt")
    root_cmake = _text("csrc/mlir/CMakeLists.txt")
    tla_dist_init = _text("catccos/tla_dist/__init__.py")
    assert not (ROOT / "catccos/tla_dist/_native.py").exists()
    assert "Python3_add_library" not in cmake
    assert "_tla_type_bridge_native" not in cmake
    assert "TlaDistCompilePipeline" in cmake
    assert '"${TLA_DIST_PYTHON_OUTPUT_DIR}/_tla_type_bridge_native*"' in root_cmake
    assert '_native = import_module("._native", __name__)' not in tla_dist_init


def test_catlass_v210_descriptor_and_environment_contracts() -> None:
    root_cmake = _text("csrc/mlir/CMakeLists.txt")
    lowering = _text("csrc/mlir/lib/Passes/TlaDistLowerCommPass.cpp")
    bitcode = _text("csrc/mlir/bc/Vector/shmem.cpp")
    build = _text("build.sh")

    assert "CATLASS_DSL_PREBUILT_ASCENDNPU_IR" in root_cmake
    assert 'set(ENV{CATLASS_DSL_ASCENDNPU_IR_ROOT}' in root_cmake
    assert "CATLASS_DSL_PREBUILT_ASCENDNPU_IR" in build
    assert "SmallVector<Value, 24> payload" in lowering
    assert "DESC_ABI_PARAMS(src), DESC_ABI_PARAMS(dst)" in bitcode
    assert "TENSOR_DESC_12(src), TENSOR_DESC_12(dst)" in bitcode
    assert "TensorDesc2D" not in bitcode
