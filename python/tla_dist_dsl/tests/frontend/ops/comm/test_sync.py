"""Parameterised synchronization frontend contracts; no device pass is implied."""
# ruff: noqa: E402
from enum import Enum
import inspect
import catccos  # noqa: F401
import pytest
pytest.importorskip("catlass._mlir.ir")
from catccos import tla_dist
from catccos.tla_dist.params import CoreScope, SignalOp, CompareOp
from catlass import tla
from catlass.execution_lowering import TlaLoweringError

DTYPES = (tla.Int8, tla.Int16, tla.Int32, tla.Int64, tla.UInt8, tla.UInt16, tla.UInt32, tla.UInt64, tla.Float32)


def slot(dtype=tla.Int32, shape=(1, 1), *, space=tla.AddressSpace.gm):
    return tla.make_fake_tensor(dtype, shape, (shape[-1], 1) if len(shape) == 2 else (1,),
        addrspace=space, origin_shape=shape, coord=(0,) * len(shape), layout_tag=tla.arch.RowMajor)


def wait_ir(dtype=tla.Int32, value=1, cmp=CompareOp.EQ, tensor=None):
    @tla.kernel
    def kernel(counter: tla.Tensor):
        with tla.vector():
            tla_dist.wait_until(counter, value, cmp=cmp)
    return kernel.dump_mlir(type_args=(slot(dtype) if tensor is None else tensor,))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("cmp", tuple(CompareOp))
def test_wait_until_matrix(dtype, cmp):
    text = wait_ir(dtype, 1.5 if dtype is tla.Float32 else 1, cmp)
    assert '"tla_dist.wait_until"' in text
    assert f"cmp = {tuple(CompareOp).index(cmp)} : i32" in text


@pytest.mark.parametrize("dtype,value", [(tla.Int8,-128),(tla.Int8,127),(tla.UInt8,255),(tla.Int16,-32768),
    (tla.UInt16,65535),(tla.Int32,-2**31),(tla.UInt32,2**32-1),(tla.Int64,-2**63),
    (tla.Int64,2**63-1),(tla.UInt64,2**63),(tla.UInt64,2**64-1),(tla.Float32,float("inf")),
    (tla.Float32,-float("inf")),(tla.Float32,3.4028234663852886e38)])
def test_wait_boundaries(dtype, value):
    assert '"tla_dist.wait_until"' in wait_ir(dtype, value)


@pytest.mark.parametrize("dtype,value", [(tla.Int8,-129),(tla.Int8,128),(tla.UInt8,-1),(tla.UInt8,256),
    (tla.UInt64,2**64),(tla.Int64,2**63),(tla.Float32,float("nan")),(tla.Float32,1e39),
    (tla.Int32,True),(tla.Float32,False),(tla.Int32,tla.UInt32(1))])
def test_wait_bad_thresholds(dtype, value):
    with pytest.raises(TlaLoweringError): wait_ir(dtype, value)


@pytest.mark.parametrize("dtype", [tla.Float16,tla.BFloat16,tla.Bool])
def test_wait_unsupported_dtype(dtype):
    with pytest.raises(TlaLoweringError, match="unsupported counter dtype"): wait_ir(dtype)


@pytest.mark.parametrize("shape", [(2,), (1,2), (2,1)])
def test_wait_rejects_non_slot(shape):
    with pytest.raises(TlaLoweringError, match="single-slot"): wait_ir(tensor=slot(shape=shape))


class Wrong(Enum):
    EQ = "eq"
    AIV = "aiv"
    SET = "set"


@pytest.mark.parametrize("bad", ["eq",0,False,Wrong.EQ,SignalOp.SET,CoreScope.AIV])
def test_compare_requires_exact_enum(bad):
    with pytest.raises(TlaLoweringError, match="CompareOp member"): wait_ir(cmp=bad)


@pytest.mark.parametrize("bad", ["aiv",0,False,Wrong.AIV,SignalOp.SET,CompareOp.EQ])
def test_scope_requires_exact_enum(bad):
    @tla.kernel
    def kernel():
        with tla.vector(): tla_dist.local_sync(scope=bad)
    with pytest.raises(TlaLoweringError, match="CoreScope member"): kernel.dump_mlir(type_args=())


@pytest.mark.parametrize("bad", ["set",0,False,Wrong.SET,CoreScope.AIV,CompareOp.EQ])
def test_signal_requires_exact_enum(bad):
    @tla.kernel
    def kernel(flag: tla.Tensor):
        with tla.vector(): tla_dist.signal(flag, 1, peer=0, op=bad)
    with pytest.raises(TlaLoweringError, match="SignalOp member"): kernel.dump_mlir(type_args=(slot(),))


@pytest.mark.parametrize("scope", tuple(CoreScope))
def test_constexpr_enum(scope):
    @tla.kernel
    def kernel(scope: tla.Constexpr[CoreScope]):
        with tla.cube(): tla.pipe_barrier(tla.pipes.ALL)
        with tla.vector():
            if tla.const_expr(scope is CoreScope.AIV): tla_dist.local_sync(scope=scope)
        if tla.const_expr(scope is CoreScope.MIX): tla_dist.local_sync(scope=scope)
    assert f"scope = {tuple(CoreScope).index(scope)} : i32" in kernel.dump_mlir(type_args=(scope,))


@pytest.mark.parametrize("name", ["local_sync","sync","barrier","fence","signal","wait_until"])
def test_exports(name):
    assert name in tla_dist.__all__
    assert callable(getattr(tla_dist,name))
    assert inspect.signature(getattr(tla_dist,name)).return_annotation in (None,"None")


def test_quiet_default_form():
    @tla.kernel
    def kernel():
        with tla.vector(): tla_dist.quiet()
    assert '"tla_dist.quiet_default"' in kernel.dump_mlir(type_args=())


@pytest.mark.parametrize("kwargs", [{"peer":0},{"event_id":2}])
def test_quiet_default_rejects_qp_arguments(kwargs):
    @tla.kernel
    def kernel():
        with tla.vector(): tla_dist.quiet(**kwargs)
    with pytest.raises(TlaLoweringError, match="default quiet does not accept"):
        kernel.dump_mlir(type_args=())
