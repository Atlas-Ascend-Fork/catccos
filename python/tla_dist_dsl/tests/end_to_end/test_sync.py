"""Standalone synchronization acceptance. Run --list before selecting device cases.

All jobs require the supervisor's external timeout and serialized device lease.
The typed writer is an independent AscendC same-width store/cache fixture; it
uses neither remote_copy nor the cancelled UInt64 copy implementation.
"""
# ruff: noqa: BLE001
import argparse
from dataclasses import dataclass
import fnmatch
import os
import struct
from catccos import tla_dist  # isort: skip
from catccos.tla_dist.params import CoreScope, CompareOp, SignalOp
from catlass import tla

DTYPES = {name: getattr(tla, name) for name in
    ("Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64", "Float32")}
FORMATS = {"Int8":"b", "Int16":"h", "Int32":"i", "Int64":"q", "UInt8":"B",
    "UInt16":"H", "UInt32":"I", "UInt64":"Q", "Float32":"f"}
WRITER_SOURCE = r'''
#include "kernel_operator.h"
#include <cstdint>
template<class T> __aicore__ void storeSlot(uint64_t addr, uint64_t bits) {
    union { uint64_t bits; T value; } value;
    value.bits = bits;
    AscendC::GlobalTensor<T> slot;
    slot.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(addr));
    slot.SetValue(0, value.value);
    __asm__ __volatile__("" ::: "memory");
    AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(slot);
    __asm__ __volatile__("" ::: "memory");
}
extern "C" __aicore__ void sync_test_write(uint64_t addr, uint64_t bits, int32_t tag, int32_t delay) {
    for (int i=0; i<delay; ++i) { __asm__ __volatile__("" ::: "memory"); }
    switch(tag) {
        case 0: storeSlot<int8_t>(addr,bits); break; case 1: storeSlot<int16_t>(addr,bits); break;
        case 2: storeSlot<int32_t>(addr,bits); break; case 3: storeSlot<int64_t>(addr,bits); break;
        case 4: storeSlot<uint8_t>(addr,bits); break; case 5: storeSlot<uint16_t>(addr,bits); break;
        case 6: storeSlot<uint32_t>(addr,bits); break; case 7: storeSlot<uint64_t>(addr,bits); break;
        case 8: storeSlot<float>(addr,bits); break;
    }
}
'''

@tla.extern(source=WRITER_SOURCE, name="sync_test_write")
def backend_write(ptr: tla.Pointer[tla.UInt8, tla.AddressSpace.gm], bits: tla.UInt64,
                  tag: tla.Int32, delay: tla.Int32) -> None: ...

@dataclass(frozen=True)
class WaitCase:
    dtype: str
    cmp: CompareOp = CompareOp.EQ
    threshold: int | float = 7
    initial: int | float = 0
    target: int | float = 7
    mixed: bool = False
    offset: bool = False
    rank1: bool = False

@tla.kernel
def typed_wait(counter: tla.Tensor, observed: tla.Tensor, rank: tla.Int32, producer: tla.Int32,
               peer: tla.Int32, index: tla.Int32, bits: tla.UInt64, delay: tla.Int32,
               case: tla.Constexpr[WaitCase]):
    if tla.const_expr(case.mixed):
        with tla.cube():
            # Actual independent cube-side progress is observable after launch.
            if tla.arch.block_idx()==0: observed[0, 1] = 2
    if tla.const_expr(case.rank1):
        counter_slot = tla.tile_view(counter, tla.make_shape(1), tla.make_coord(index))
    else:
        counter_slot = tla.tile_view(counter, tla.make_shape(1,1), tla.make_coord(0,index))
    with tla.vector():
        if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
            if rank == producer:
                pointer = tla_dist.shmem_ptr(counter.ptr + index, peer)
                backend_write(tla.recast_ptr(pointer, dtype=tla.UInt8), bits,
                    tuple(DTYPES).index(case.dtype), delay)
            else:
                tla_dist.wait_until(counter_slot, case.threshold, cmp=case.cmp)
                observed[0,0] = 1

@tla.kernel
def collective(observed: tla.Tensor, case: tla.Constexpr[str], team: tla.Int32):
    if tla.const_expr(case.startswith("mix")):
        with tla.cube():
            if tla.arch.block_idx()==0: observed[0,1] = 2
    with tla.vector():
        if tla.const_expr(case.startswith("mix")):
            flat = tla.arch.block_idx() * 2 + tla.arch.sub_block_idx()
        else:
            flat = tla.arch.block_idx()
        observed[0,flat + 2] = 1
        if tla.const_expr(not case.startswith("mix")):
            if tla.const_expr(case.endswith("local")): tla_dist.local_sync()
            elif tla.const_expr(case.endswith("barrier")): tla_dist.barrier(team=team)
            else: tla_dist.sync(team=team)
    if tla.const_expr(case.startswith("mix")):
        if tla.const_expr(case.endswith("local")): tla_dist.local_sync(scope=CoreScope.MIX)
        elif tla.const_expr(case.endswith("barrier")): tla_dist.barrier(team=team,scope=CoreScope.MIX)
        else: tla_dist.sync(team=team,scope=CoreScope.MIX)
    with tla.vector():
        if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
            observed[0,0] = 3

@tla.kernel
def signal_case(flag: tla.Tensor, peer: tla.Int32, value: tla.Int32, add: tla.Constexpr[bool]):
    slot = tla.tile_view(flag, tla.make_shape(1,1), tla.make_coord(0,0))
    with tla.vector():
        if tla.const_expr(add):
            tla_dist.signal(slot, value, peer=peer,op=SignalOp.ADD)
        else:
            if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
                tla_dist.signal(slot, value, peer=peer)
        tla_dist.barrier()

@tla.kernel
def publication(payload: tla.Tensor, source: tla.Tensor, flag: tla.Tensor, ack: tla.Tensor,
                output: tla.Tensor, peer: tla.Int32, rounds: tla.Constexpr[int]):
    scratch = tla.make_tensor(tla.allocate(16,tla.Float32,tla.AddressSpace.ub,32),
        tla.make_layout(tla.make_shape(1,16),tla.make_stride(16,1),layoutTag=tla.arch.RowMajor))
    ready = tla.tile_view(flag,tla.make_shape(1,1),tla.make_coord(0,0))
    acknowledged = tla.tile_view(ack,tla.make_shape(1,1),tla.make_coord(0,0))
    completed = tla.flag("payload_complete",tla.arch.MTE3,tla.arch.MTE2,id=2)
    submitted = tla.flag("payload_read",tla.arch.MTE2,tla.arch.MTE3,id=2)
    with tla.vector():
        for epoch in tla.range(1,rounds+1):
            if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
                row = tla.tile_view(source,tla.make_shape(1,16),tla.make_coord(epoch-1,0))
                tla_dist.remote_copy(payload,row,tla_dist.params.CopyParams(mode="put"),peer=peer,
                    ub_tensor=scratch,event_id=2)
                tla.set_flag(completed); tla.wait_flag(completed)
                tla_dist.signal(ready,epoch,peer=peer)
                tla_dist.wait_until(ready,epoch,cmp=CompareOp.EQ)
                # MTE2 starts only after wait. Its explicit completion precedes MTE3.
                tla.copy(scratch,payload)
                tla.set_flag(submitted); tla.wait_flag(submitted)
                out = tla.tile_view(output,tla.make_shape(1,16),tla.make_coord(epoch-1,0))
                tla.copy(out,scratch)
                tla.set_flag(completed); tla.wait_flag(completed)
                tla_dist.signal(acknowledged,epoch,peer=peer)
                tla_dist.wait_until(acknowledged,epoch,cmp=CompareOp.EQ)
            tla_dist.local_sync()


@tla.kernel
def default_ordering(observed:tla.Tensor):
    with tla.vector():
        if tla.arch.block_idx()==0 and tla.arch.sub_block_idx()==0:
            tla_dist.fence()
            tla_dist.quiet()
            observed[0,0]=9

@tla.kernel
def ub_guard(flag:tla.Tensor,observed:tla.Tensor,peer:tla.Int32):
    ptr=tla.allocate(188*1024,tla.UInt8,tla.AddressSpace.ub,32)
    storage=tla.make_tensor(ptr,tla.make_layout(tla.make_shape(1,188*1024),tla.make_stride(188*1024,1),layoutTag=tla.arch.RowMajor))
    slot=tla.tile_view(flag,tla.make_shape(1,1),tla.make_coord(0,0))
    with tla.vector():
        if tla.arch.block_idx()==0 and tla.arch.sub_block_idx()==0:
            storage[0,0]=tla.UInt8(17)
            storage[0,188*1024-1]=tla.UInt8(29)
            tla_dist.signal(slot,1,peer=peer,op=SignalOp.ADD)
            observed[0,0]=tla.Int32(storage[0,0])
            observed[0,1]=tla.Int32(storage[0,188*1024-1])
        tla_dist.barrier()


def manifest():
    for blocks in (1,2,4): yield f"p0/vector-local/blocks-{blocks}", ("collective","aiv-local",0,blocks)
    yield "p0/vector-local/resident-limit", ("collective","aiv-local",0,0)
    for scope in ("aiv","mix"):
        for kind in ("local","sync","barrier"):
            yield f"collective/{scope}-{kind}", ("collective",f"{scope}-{kind}",0,2)
        yield f"collective/{scope}-invalid", ("collective",f"{scope}-barrier",-1,2)
    for dtype in DTYPES:
        for cmp in CompareOp:
            threshold = {CompareOp.EQ:7,CompareOp.NE:6,CompareOp.GT:6,CompareOp.GE:7,CompareOp.LT:8,CompareOp.LE:7}[cmp]
            initial = {CompareOp.EQ:0,CompareOp.NE:6,CompareOp.GT:0,CompareOp.GE:0,CompareOp.LT:9,CompareOp.LE:9}[cmp]
            yield f"typed/{dtype}/{cmp.name}", ("wait", WaitCase(dtype,cmp,threshold,initial,7))
        width=DTYPES[dtype].width
        boundary = 2**width-1 if dtype.startswith("UInt") else (-2**(width-1) if dtype.startswith("Int") else float("inf"))
        initial = 0 if boundary != 0 else 1
        yield f"boundary/{dtype}", ("wait",WaitCase(dtype,threshold=boundary,initial=initial,target=boundary))
        yield f"address/{dtype}", ("wait",WaitCase(dtype,offset=True,rank1=True))
    yield "typed/mixed-width", ("wait",WaitCase("UInt64",threshold=2**63,initial=0,target=2**63,mixed=True))
    yield "signal/set", ("signal",False,1)
    yield "signal/add", ("signal",True,1)
    yield "signal/add-multiwriter", ("signal",True,4)
    yield "ordering/publication-epoch-ack", ("publication",100)
    yield "ordering/default-fence-quiet", ("default",)
    yield "resource/user-ub-guards", ("ub-guard",)



def _view(tensor): return tla_dist.from_dlpack(tensor,layout_tag=tla.arch.RowMajor)


def _compile(runtime,kernel,args,*,mixed=False):
    artifact,error=None,None
    try:
        artifact=tla_dist.compile(kernel,*args,arch_scope="aic.c310" if mixed else "aiv.c310",cache_dir=runtime.cache_dir)
    except Exception as exc: error=f"compile: {type(exc).__name__}: {exc}"
    runtime.check_errors(error)
    return artifact


def _launch(runtime,artifact,args,blocks,*,local_only=False):
    runtime.torch.npu.synchronize(); runtime.shmem.barrier_all()
    error=None
    try:
        if not local_only or runtime.rank==0:
            artifact(*args,block_num=blocks)
            runtime.torch.npu.synchronize()
    except Exception as exc: error=f"launch: {type(exc).__name__}: {exc}"
    runtime.check_errors(error)


def _wait(runtime,case):
    torch=runtime.torch
    name=case.dtype
    fmt=FORMATS[name]; size=struct.calcsize(fmt)
    raw=runtime.shmem.symmetric_tensor((1,256),torch.uint8)
    output=torch.zeros((1,4),dtype=torch.int32,device=raw.device)
    # Torch's byte view constructs metadata only; no UInt64 NPU arithmetic or copy is required.
    typed=raw.view(getattr(torch,{"Float32":"float32"}.get(name,name.lower())))
    counter=typed[:,4:] if case.offset else typed
    if case.rank1: counter=counter.reshape(-1)
    index=3 if case.offset else 0
    byte_offset=(4+index)*size if case.offset else 0
    artifact=None
    try:
        for producer in (0,1):
            for delayed in (False,True):
                expected=bytearray([0xA5]*256)
                expected[byte_offset:byte_offset+size]=struct.pack("<"+fmt,case.initial)
                raw.copy_(torch.tensor(list(expected),dtype=torch.uint8).reshape(1,256))
                output.zero_()
                bits=int.from_bytes(struct.pack("<"+fmt,case.target),"little")
                args=(_view(counter),_view(output),tla.Int32(runtime.rank),tla.Int32(producer),
                    tla.Int32(1-runtime.rank),tla.Int32(index),tla.UInt64(bits),tla.Int32(1000000 if delayed else 0))
                if artifact is None: artifact=_compile(runtime,typed_wait,(*args,case),mixed=case.mixed)
                _launch(runtime,artifact,args,1)
                if runtime.rank != producer:
                    expected[byte_offset:byte_offset+size]=struct.pack("<"+fmt,case.target)
                actual=bytes(raw.cpu().reshape(-1).tolist())
                error=None
                if actual != bytes(expected): error=f"typed writer/address/guard mismatch {name}/{case.cmp.name}"
                seen=output.cpu()[0].tolist()
                if seen[0] != (1 if runtime.rank != producer else 0): error=f"wait return marker mismatch {seen}"
                if case.mixed and seen[1] != 2: error=f"cube progress missing {seen}"
                runtime.check_errors(error)
                if runtime.rank == 0: print(f"DETAIL producer={producer} delayed={delayed}",flush=True)
    finally: runtime.shmem.free(raw)


def _run(runtime,item,*,local_only=False):
    torch=runtime.torch
    if item[0]=="wait": return _wait(runtime,item[1])
    if item[0]=="collective":
        _,case,team,blocks=item
        if blocks==0:
            from catccos.tla_dist._sync_runtime import resident_block_limit
            blocks=resident_block_limit(int(os.environ["LOCAL_RANK"]),paired=False)
        observed=torch.zeros((1,128),dtype=torch.int32,device=f"npu:{os.environ['LOCAL_RANK']}")
        args=(_view(observed),case,tla.Int32(team))
        artifact=_compile(runtime,collective,args,mixed=case.startswith("mix"))
        _launch(runtime,artifact,(args[0],args[2]),blocks,local_only=local_only)
        actual=observed.cpu()[0].tolist()
        expected=blocks*(2 if case.startswith("mix") else 1)
        runtime.check_errors(None if (local_only and runtime.rank!=0) or (actual[0]==3 and sum(actual[2:])==expected) else f"participant markers {actual}")
    elif item[0]=="signal":
        _,add,blocks=item
        flag=runtime.shmem.symmetric_tensor((1,16),torch.int32)
        try:
            flag.zero_(); args=(_view(flag),tla.Int32(1-runtime.rank),tla.Int32(3),add)
            artifact=_compile(runtime,signal_case,args)
            _launch(runtime,artifact,args[:-1],blocks)
            runtime.check_errors(None if flag.cpu()[0,0].item()==3*(blocks if add else 1) else "signal result mismatch")
        finally: runtime.shmem.free(flag)
    elif item[0]=="default":
        output=torch.zeros((1,4),dtype=torch.int32,device=f"npu:{os.environ['LOCAL_RANK']}")
        args=(_view(output),)
        artifact=_compile(runtime,default_ordering,args)
        _launch(runtime,artifact,args,1)
        runtime.check_errors(None if output.cpu()[0,0].item()==9 else "default completion marker missing")
    elif item[0]=="ub-guard":
        flag=runtime.shmem.symmetric_tensor((1,16),torch.int32)
        try:
            flag.zero_()
            output=torch.zeros((1,4),dtype=torch.int32,device=flag.device)
            args=(_view(flag),_view(output),tla.Int32(1-runtime.rank))
            artifact=_compile(runtime,ub_guard,args)
            _launch(runtime,artifact,args,1)
            seen=output.cpu()[0].tolist()
            runtime.check_errors(None if seen[:2]==[17,29] and flag.cpu()[0,0].item()==1 else f"UB guard/signal mismatch {seen}")
        finally: runtime.shmem.free(flag)
    else:
        rounds=item[1]
        payload=runtime.shmem.symmetric_tensor((1,16),torch.float32)
        flag=runtime.shmem.symmetric_tensor((1,16),torch.int32)
        ack=runtime.shmem.symmetric_tensor((1,16),torch.int32)
        try:
            payload.fill_(-1);flag.zero_();ack.zero_()
            source=(torch.arange(rounds*16,dtype=torch.float32)+runtime.rank*1024).reshape(rounds,16).to(payload.device)
            output=torch.full_like(source,-1)
            args=tuple(_view(t) for t in (payload,source,flag,ack,output))+(tla.Int32(1-runtime.rank),rounds)
            artifact=_compile(runtime,publication,args)
            _launch(runtime,artifact,args[:-1],2)
            runtime.assert_equal(output,(torch.arange(rounds*16,dtype=torch.float32)+(1-runtime.rank)*1024).reshape(rounds,16))
        finally:
            for tensor in (ack,flag,payload): runtime.shmem.free(tensor)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list",action="store_true")
    parser.add_argument("--case",default="*",help="glob over manifest names")
    args=parser.parse_args();cases=[(name,item) for name,item in manifest() if fnmatch.fnmatch(name,args.case)]
    if args.list:
        for name,_ in cases: print(name)
        return 0
    if not cases: raise SystemExit("no matching acceptance cases")
    if os.environ.get("WORLD_SIZE")!="2": raise SystemExit("use torchrun --standalone --nproc-per-node=2")
    from test_remote_copy_mte import initialized_runtime
    with initialized_runtime() as runtime:
        for name,item in cases:
            if runtime.rank==0: print(f"RUN  {name}",flush=True)
            local_only=name.startswith("p0/")
            _run(runtime,item,local_only=local_only)
            if not local_only or runtime.rank==0:
                print(f"DEVICE_PASS {name} rank={runtime.rank} launches={4 if item[0]=='wait' else 1}",flush=True)
            runtime.dist.barrier()
            if runtime.rank==0: print(f"PASS {name}",flush=True)
        if runtime.rank==0: print(f"{len(cases)} passed",flush=True)
    return 0

if __name__=="__main__": raise SystemExit(main())
