"""Native verification and device-link checks for synchronization kernels."""

import subprocess

import pytest

import catccos  # noqa: F401

pytest.importorskip("catlass._mlir.ir")
from catlass import tla
from frontend.ops.comm.test_sync import DTYPES, slot

from catccos import tla_dist
from catccos.tla_dist.compiler import _execution
from catccos.tla_dist.compiler_bridge import resolve_tla_dist_compile
from catccos.tla_dist.params import CompareOp, CoreScope


@tla.kernel
def wait_kernel(counter: tla.Tensor, threshold, mode: tla.Constexpr[str]):
    if tla.const_expr(mode == "mix"):
        with tla.cube():
            tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        tla_dist.wait_until(counter, threshold, cmp=CompareOp.GE)
        if tla.const_expr(mode == "mix_aiv"):
            tla_dist.local_sync()


@tla.kernel
def collective_kernel(mode: tla.Constexpr[str], op: tla.Constexpr[str]):
    if tla.const_expr(mode == "mix"):
        with tla.cube():
            tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        if tla.const_expr(mode != "mix"):
            if tla.const_expr(op == "local"):
                tla_dist.local_sync()
            elif tla.const_expr(op == "sync"):
                tla_dist.sync()
            elif tla.const_expr(op == "invalid"):
                tla_dist.barrier(team=-1)
            else:
                tla_dist.barrier()
    if tla.const_expr(mode == "mix"):
        if tla.const_expr(op == "local"):
            tla_dist.local_sync(scope=CoreScope.MIX)
        elif tla.const_expr(op == "sync"):
            tla_dist.sync(scope=CoreScope.MIX)
        else:
            tla_dist.barrier(scope=CoreScope.MIX)


@tla.kernel
def default_kernel(flag: tla.Tensor, op: tla.Constexpr[str]):
    with tla.vector():
        if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
            if tla.const_expr(op == "quiet"):
                tla_dist.quiet()
            elif tla.const_expr(op == "fence"):
                tla_dist.fence()
            elif tla.const_expr(op == "set"):
                tla_dist.signal(flag, 1, peer=0)
            else:
                tla_dist.signal(flag, 1, peer=0, op=tla_dist.SignalOp.ADD)


def require_native():
    compiler = resolve_tla_dist_compile()
    if compiler is None:
        pytest.skip("build TlaDistCompile for native verification")
    return compiler


def lower(tmp_path, text, flags=()):
    source = tmp_path / "sync.mlir"
    source.write_text(text)
    return subprocess.run(
        [str(require_native()), str(source), *flags, "-o", "-"],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", ["aiv", "mix", "mix_aiv"])
def test_sync_kernel_links(tmp_path, dtype, mode):
    require_native()
    if _execution._find_hivmc_a5() is None:
        pytest.skip("configure CANN for device linkage")
    artifact = tla_dist.compile(
        wait_kernel,
        type_args=(slot(dtype), dtype(1), mode),
        arch_scope="aic.c310" if mode == "mix" else "aiv.c310",
        cache_dir=tmp_path,
    )
    assert artifact.kernel_binary_path.is_file()
    assert "tla_dist_wait_" in artifact.artifacts.LLVM


@pytest.mark.parametrize(
    "mode,op",
    [(m, o) for m in ("aiv", "mix") for o in ("local", "sync", "barrier")]
    + [("aiv", "invalid")],
)
def test_collective_kernel_links(tmp_path, mode, op):
    require_native()
    if _execution._find_hivmc_a5() is None:
        pytest.skip("configure CANN for device linkage")
    artifact = tla_dist.compile(
        collective_kernel,
        type_args=(mode, op),
        arch_scope="aic.c310" if mode == "mix" else "aiv.c310",
        cache_dir=tmp_path,
    )
    assert artifact.kernel_binary_path.is_file()
    text = artifact.artifacts.LLVM
    from catccos.tla_dist.compiler import SyncCompiledFunction

    assert isinstance(artifact, SyncCompiledFunction)
    if mode == "mix":
        assert "tla_dist_" + ({"local": "local_sync"}.get(op, op)) + "_mix_aic" in text
        assert "tla_dist_" + ({"local": "local_sync"}.get(op, op)) + "_mix_aiv" in text
    else:
        assert "hivm.core_ratio<0, 1>" in text
        assert "tla_dist.vector_hard_sync" in text
        assert "_mix_aic" not in text


@pytest.mark.parametrize("op", ["quiet", "fence", "set", "add"])
def test_default_helper_kernel_links(tmp_path, op):
    require_native()
    if _execution._find_hivmc_a5() is None:
        pytest.skip("configure CANN for device linkage")
    artifact = tla_dist.compile(
        default_kernel, type_args=(slot(), op), cache_dir=tmp_path
    )
    assert artifact.kernel_binary_path.is_file()


@pytest.mark.parametrize("attribute", ["7 : i32", "0 : i64", '"eq"'])
def test_closed_enum_ir_rejects_invalid_encoding(tmp_path, attribute):
    text = wait_kernel.dump_mlir(type_args=(slot(), tla.Int32(1), "aiv"))
    text = text.replace("cmp = 3 : i32", f"cmp = {attribute}")
    result = lower(tmp_path, text)
    assert result.returncode != 0
    assert "cmp" in result.stderr


def test_mix_split_pairs_and_attributes(tmp_path):
    result = lower(tmp_path, collective_kernel.dump_mlir(type_args=("mix", "sync")))
    assert result.returncode == 0, result.stderr
    text = result.stdout
    for side in ("aic", "aiv"):
        assert f"tla_dist_sync_mix_{side}" in text
    assert "convergent" in text
    assert "readnone" not in text
    assert "willreturn" not in text


def test_collective_dynamic_control_rejected(tmp_path):
    @tla.kernel
    def kernel(value: tla.Int32):
        with tla.vector():
            if value == 0:
                tla_dist.local_sync()

    result = lower(tmp_path, kernel.dump_mlir(type_args=(tla.Int32(0),)))
    assert result.returncode != 0
    assert "dynamic if/while" in result.stderr


def test_default_event_zero_collision_rejected(tmp_path):
    @tla.kernel
    def kernel():
        event = tla.flag("user", tla.arch.MTE2, tla.arch.MTE3, id=0)
        with tla.vector():
            tla.set_flag(event)
            tla.wait_flag(event)
            tla_dist.sync()

    result = lower(tmp_path, kernel.dump_mlir(type_args=()))
    assert result.returncode != 0
    assert "event 0" in result.stderr


def test_default_resources_preserve_user_ub_limit(tmp_path):
    result = lower(tmp_path, collective_kernel.dump_mlir(type_args=("aiv", "barrier")))
    assert result.returncode == 0, result.stderr
    assert "tla.ub_programmable_bytes = 192512" in result.stdout
    assert "tla_dist.sync_resource_abi = 1" in result.stdout


def test_statically_nonowner_default_quiet_is_rejected(tmp_path):
    @tla.kernel
    def kernel():
        with tla.vector():
            if tla.arch.block_idx() != 0:
                tla_dist.quiet()

    result = lower(tmp_path, kernel.dump_mlir(type_args=()))
    assert result.returncode != 0
    assert "flat AIV0 owner" in result.stderr


def test_static_ub_overflow_is_rejected(tmp_path):
    @tla.kernel
    def kernel():
        tla.allocate(188 * 1024 + 32, tla.UInt8, tla.AddressSpace.ub, 32)
        with tla.vector():
            tla_dist.sync()

    result = lower(tmp_path, kernel.dump_mlir(type_args=()))
    assert result.returncode != 0
    assert "188 KiB" in result.stderr


def test_mixed_wait_types_share_one_kernel(tmp_path):
    require_native()
    if _execution._find_hivmc_a5() is None:
        pytest.skip("configure CANN for device linkage")

    @tla.kernel
    def kernel(counter: tla.Tensor, other: tla.Tensor, value: tla.UInt64):
        with tla.vector():
            tla_dist.wait_until(counter, value)
            tla_dist.wait_until(other, 1)

    artifact = tla_dist.compile(
        kernel,
        type_args=(slot(tla.UInt64), slot(tla.Int32), tla.UInt64(1)),
        cache_dir=tmp_path,
    )
    assert artifact.kernel_binary_path.is_file()
    assert "tla_dist_wait_u64" in artifact.artifacts.LLVM
    assert "tla_dist_wait_i32" in artifact.artifacts.LLVM


def test_handwritten_ir_team_upper_bound_rejected(tmp_path):
    text = collective_kernel.dump_mlir(type_args=("aiv", "sync"))
    # The constant is an SSA operand, not an enum attribute.
    text = text.replace("value = 0 : i32", "value = 2048 : i32")
    result = lower(tmp_path, text)
    assert result.returncode != 0
    assert "team constant" in result.stderr
