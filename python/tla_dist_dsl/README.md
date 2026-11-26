# CATCCOS TLA_DIST DSL

This minimal extension provides one end-to-end example for Ascend 950PR&950DT products:
`examples/fusion_ops/matmul_reduce_scatter.py`. For rank `r`, its output is
`sum(A_p @ B_p for p in ranks)[r * (M / world_size):(r + 1) * (M / world_size), :]`.
Inputs use matching FP16 or BF16 dtypes and row-major or column-major layouts;
output and communication use FP32. M must be divisible by the rank count.

Compute operations and types come from the pinned `catlass.tla` package.
CATCCOS owns the `tla_dist` MLIR dialect, communication lowering, SHMEM device
bitcode, compiler/launch adapter, and SHMEM host lifecycle. The communication
surface consists of:

- `remote_copy(dst, src, params=None, *, peer, ub_tensor=None, event_id=None)`:
  asynchronously copy matching RowMajor FP32 tensors. UInt64 additionally
  supports non-atomic MTE GM→GM GET/PUT with matching UInt64 UB scratch;
  signed Int64, UInt64 GM↔UB, other transports and UInt64 atomic ADD are not
  enabled. GET resolves `src` on
  the peer; PUT resolves `dst` on the peer. Source and destination column
  strides must be provably 1; row padding, dynamic shapes and dynamic row
  strides are supported. Dynamic column strides that cannot be proven to be
  1 are rejected during device lowering. MTE supports GM→GM and GM↔UB;
  GM→GM requires an explicit `CopyParams(mode="get")` or `CopyParams(mode="put")`.
  For GM↔UB, `mode` may be omitted (`None`): GM→UB selects GET and UB→GM
  selects PUT. Explicit GET/PUT values that conflict with a GM↔UB route raise
  an error. Atomic ADD requires MTE and a GM destination.
  `CopyParams` adds `mode`, `transport`, `atomic_mode`, the UDMA-only
  `wqe_pipe`, and `qp_idx` to upstream compute parameters. `qp_idx` defaults
  to 0 and selects the RDMA/UDMA queue. It accepts a constant or runtime DSL
  integer; MTE ignores it.
- `barrier_all()`: synchronize participating ranks and AIVs.
- `quiet(params, *, peer, ub_tensor=None, event_id=None)`: wait for earlier
  RDMA/UDMA GET/PUT requests to the peer on `params.transport` and
  `params.qp_idx`. `params` must be a `CopyParams` object; only these two
  fields are used. `mode`, `atomic_mode`, and `wqe_pipe` do not affect the wait.
  RDMA requires UB scratch and a compile-time integer `event_id`; UDMA accepts neither.

RDMA and UDMA support GM→GM GET/PUT on AIV and mixed-AIV kernels with explicit
synchronization. RDMA availability and backend settings are read from the selected
SHMEM build configuration. SDMA device compilation reports `device lowering is not implemented`.

RDMA/UDMA split large contiguous segments into NBI requests. RDMA requests
contain at most `UINT32_MAX / sizeof(float)` elements (4 GiB minus 4 B),
matching the XSCALE/HNS_1825 WQE byte-length field; UDMA requests contain at
most 256 MiB. Contiguous tiles are merged before splitting; padded tiles
are split within each row without copying padding. The helper adds no quiet;
the caller waits for all submitted chunks through `quiet`.

| Route / submission | UB scratch | Reserved directions for `event_id` |
| --- | --- | --- |
| MTE GM→GM | Matching FP32 or UInt64 staging; rows padded to 32 B | MTE2→MTE3 |
| MTE GM↔UB | None | None |
| RDMA GM→GM / quiet, XSCALE | 128 B control workspace | SCALAR→MTE3, MTE3→SCALAR |
| RDMA GM→GM / quiet, HNS_1825 | 128 B control workspace | SCALAR→MTE3, MTE3→SCALAR, MTE2→SCALAR |
| UDMA GM→GM, `wqe_pipe="mte3"` (default) | 256 B WQE workspace | SCALAR→MTE3, MTE3→SCALAR |
| UDMA GM→GM, `wqe_pipe="s"`; UDMA quiet | None | None |

`event_id` is a compile-time integer in `[0, 8)`. Supply it together with UB
scratch on routes that require them; omit both on the other routes. Control
workspaces may use UInt8. RDMA/UDMA lowering checks the scratch tile's 32 B
alignment and contiguous capacity; the caller manages its lifetime.

Immediately before CATLASS flag lowering, CATCCOS reserves the specified ID in
every direction listed above by inserting unused `tla.flag` declarations.
CATLASS's existing allocator skips these IDs for automatic flags, then removes
the declarations without emitting set/wait instructions. Communication lowering
passes the integer ID directly to the device helper. Explicit reuse of the same
direction and ID remains the caller's responsibility; CATCCOS does not check it.

For MTE GM→GM copies, the helper uses the ID to synchronize MTE2→MTE3 staging
through UB. The caller chooses separate IDs for double-buffered UB slots and
uses ordinary MTE3→MTE2 flags to wait before reusing each slot. RDMA/UDMA
submissions on the same AIV may reuse one ID: submission completes its local
pipeline synchronization before returning, while the GM data transfer may
remain in flight until `quiet` completes.

All communication operations require a `tla.vector()` region. For example:

```python
copy_done = tla.flag("copy_done", tla.arch.MTE3, tla.arch.MTE2)
with tla.vector():
    tla_dist.remote_copy(
        dst, src,
        tla_dist.params.CopyParams(mode="get", transport="mte",
                                   atomic_mode=tla_dist.params.AtomicMode.ADD),
        peer=peer, ub_tensor=scratch, event_id=0,
    )
    tla.set_flag(copy_done)
    tla.wait_flag(copy_done)
```

Returning from `remote_copy` does not imply completion. The caller arranges
completion waits, scratch reuse, remote readiness and cross-rank visibility.
The MTE GM→GM helper only synchronizes its internal MTE2→MTE3 staging edge;
it does not allocate another event for the reverse direction or wait for
the final store. For an 8×13 FP32 tile, staging needs 8×16×4 = 512 B.
For UInt64, round each scratch row to a multiple of four 8-byte elements:
an 8×13 tile needs 8×16×8 = 1024 B. The payload still contains 13 elements
per row, and source/destination padding is not copied. GM offsets and row
strides are in elements of the selected dtype.
RDMA/UDMA callers use `quiet` before consuming GET results, reusing payload
buffers, or notifying a PUT receiver.

Use `catccos.tla_dist.compile`
to lower communication operations and launch with `block_num=<AIC count>`.
Supply kernel arguments either positionally/by keyword or through explicit
`type_args`, without mixing the two forms. Compile options such as `cache` and
`options` can accompany either form.
The example retains two-stage workspace and compute buffers, AIC/AIV
handshakes, M/N/K tails, dynamic tensor arguments, and tiling/swizzle parameters
from the reference branch. Parameters are in `examples/common/params.py`.

## Dependencies and build

From the CATCCOS repository root, initialize the pinned sources:

```bash
git submodule update --init 3rdparty/catlass 3rdparty/shmem
git -C 3rdparty/catlass submodule update --init --recursive python/tla_dsl/3rdparty/AscendNPU-IR
```

Use the Catlass and SHMEM revisions pinned by this repository, and the
AscendNPU-IR revision pinned by Catlass. Any prebuilt AscendNPU-IR
installation must match that pinned revision.

`build.sh` applies the reviewed [Catlass UInt64 UB constructor source patch](tools/patches/README.md)
before building Catlass. The gitlink remains pinned; the patch only permits
UInt64 UB tensor construction for communication scratch, without enabling
signed Int64, other address spaces, arithmetic, or Catlass local-copy routes.
The patch helper validates the pinned revision, patch SHA256, exact source
before/after hashes, and absence of unrelated tracked changes. It is
idempotent; incompatible source or patch drift stops the build. Record the
Catlass revision together with the emitted patch and source hashes when
reporting a build or device-test result.

SHMEM device headers, Python package and runtime library must use the same
revision; build/install the pinned SHMEM package using its own instructions.
NPU execution also requires compatible PyTorch and `torch_npu` installations.

Build the checked-out AscendNPU-IR following Catlass's
[build guide](https://gitcode.com/cann/catlass/blob/76946bb3feb6efef4efb204a74fff44196b73c14/python/tla_dsl/docs/zh/dsl_development/build_guide/ascend_npu_ir.md).
Apply the pinned Catlass patches once before configuring it. From the CATCCOS repository root:

```bash
export CATLASS_DSL_ASCENDNPU_IR_ROOT="$PWD/3rdparty/catlass/python/tla_dsl/3rdparty/AscendNPU-IR"
bash 3rdparty/catlass/python/tla_dsl/tools/patches/apply_patch.sh
```

The visibility patch removes `-fvisibility=hidden` so Catlass's native Python
bridge can resolve LLVM/MLIR symbols. Existing AscendNPU-IR builds configured
with that flag must be reconfigured, rebuilt and installed after patching.

From `python/tla_dist_dsl`, with the CANN environment activated:

```bash
python -m pip install "pybind11==2.13.6"
export ASCEND_HOME_PATH=/path/to/cann
export BISHENG_COMPILER_PATH=/path/to/bisheng/bin
bash build.sh
python -m pip install -e .
```

CATCCOS reads RDMA settings from the selected SHMEM source tree. The default
is `3rdparty/shmem`; select a third-party SHMEM tree with:

```bash
TLA_DIST_DSL_SHMEM_ROOT=/path/to/shmem bash build.sh
```

The selected tree must contain the device sources. If `build/CMakeCache.txt`
or `build/compile_commands.json` is missing, CATCCOS prints the missing path
and builds with RDMA disabled; MTE and UDMA remain available. UDMA helpers
are compiled with relay disabled and require a matching SHMEM build at runtime.
To enable RDMA, build SHMEM first and use its matching Host library and Python
package at runtime.
CATCCOS reads `ACLSHMEM_RDMA_SUPPORT`, `SOC_TYPE`, and `ACLSHMEM_RDMA_BACKEND`
from the cache and checks the device compilation definitions against them.
With RDMA enabled, the supported configurations are Ascend950 with XSCALE or
HNS_1825. HNS_1825's effective `ACLSHMEMI_HNS_1825_PI_ON_CHIP=0` or `=1`
is also read from the device commands and applied to both AIV and mixed-AIV
helpers. Invalid or inconsistent configuration causes a configuration error.

When SHMEM has `ACLSHMEM_RDMA_SUPPORT=OFF`, CATCCOS omits RDMA GET/PUT and
quiet helpers. Lowering these operations reports `RDMA is not enabled in this
build`. CMake tracks both SHMEM configuration files and reruns configuration
when they change. `TLA_DIST_RDMA_BACKEND` is an internal detected value.

`build.sh` selects pybind11 from the active Python package for both CMake
builds, avoiding a conflicting Conda prefix installation. It first
builds/installs Catlass's bundled `catlass._mlir` and `_tla_type_bridge_native.so`, then
builds `TlaDistCompile` and the communication bitcode. Compute templates are
generated with the pinned Catlass's `bc_compile` helpers during the CMake build. CATCCOS does not load a
second native MLIR bridge into Python; it exchanges MLIR text with the compiler
subprocess. The extension uses private interfaces of the pinned Catlass DSL.

Install the SHMEM Python package from the current submodule.
`ShmemRuntime(transport="rdma"/"udma", qp_num=1)` converts UID bytes with
`UniqueId.from_bytes()` and creates an `InitAttr`. It calls the public Python
bindings `aclshmemx_set_attr_uniqueid_args` and `aclshmemx_set_qp_num`, then
sets `option_attr.data_op_engine_type` and calls `aclshmemx_init_attr`.
The default MTE path keeps SHMEM's four-argument Python UID initializer and
ignores `qp_num`. Host and stream barriers also use SHMEM's public Python
bindings. For RDMA/UDMA, all ranks must use the same QP count, and kernels
must keep `0 <= qp_idx < qp_num`. Each `(peer, qp_idx)` is submitted and waited
on sequentially by one AIV. UDMA QP APIs require `ACLSHMEM_RELAY_SUPPORT=OFF`;
the build checks the available SHMEM configuration for this requirement.
Older installed wheels must be replaced when updating the submodule.

The build emits separate AIV and mixed-AIV templates. The mixed template uses
`-D__CCE_AICORE_ENABLE_MIX__=1` to select SHMEM hardware synchronization through
`AscendC::SyncAll<true>()` while retaining ordinary AIV code generation and
plain C ABI exports. The final HIVMC invocation performs mixed-kernel code
generation. Rebuild after changing device functions; `--force-recompile` only
recompiles the example kernel.

The root CMake build also supports `-DCATCCOS_ENABLE_TLA_DIST_DSL=ON` and target
`tla_dist-compiler`. CMake accepts `CATLASS_TLA_DSL_ROOT` and
`TLA_DIST_DSL_SHMEM_ROOT` for compatible external source trees. `build.sh` uses
the pinned Catlass submodule. A shared `CATLASS_DSL_PREBUILT_ASCENDNPU_IR`
override is available for a matching external build and forwarded to Catlass's source/install settings; these can also be
set individually with `CATLASS_DSL_ASCENDNPU_IR_ROOT` and
`CATLASS_DSL_ASCENDNPU_IR_INSTALL_DIR`. Set `TLA_DIST_DSL_COMPILE` for an external compiler
executable or `TLA_DIST_DSL_BASE_BITCODE` for custom templates matching the kernel
mode. The nested editable install defaults to `PIP_NO_BUILD_ISOLATION=false`.

## Run the example

From `python/tla_dist_dsl`:

```bash
# Generate frontend IR without initializing a device or SHMEM.
python examples/fusion_ops/matmul_reduce_scatter.py --dump-tlair

# Two local ranks on physical devices 2 and 5.
ASCEND_RT_VISIBLE_DEVICES=2,5 torchrun --standalone --nproc-per-node=2 \
  examples/fusion_ops/matmul_reduce_scatter.py --force-recompile
```

Use `--help` for shape, layout, dtype, rank, core count and cache options.
Default dimensions `(M, N, K) = (2060, 2516, 166)` exercise tail tiles and
multiple communication rounds. The example compares the output shard with a
CPU reference. All ranks must allocate symmetric memory in the same order and
with identical sizes, and participating AIVs must execute matching barriers.
Torch owns the ACL lifecycle; SHMEM is finalized before the process group.

## Tests

With native MLIR Python bindings available:

```bash
python -m pip install -e '.[test]'
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python -m pytest -q
cmake --build csrc/mlir/build --target check-tla-dist-lit
```

Pytest automatically skips RDMA kernel compilation and linking tests when the
build disables RDMA; they run when the selected SHMEM build enables XSCALE or
HNS_1825. The tests read `TlaDistCompile.json` beside the selected compiler,
which CMake generates and packages with the executable. Frontend validation
and build script tests run regardless of the selected backend.
Lit also selects tests using the build configuration: an RDMA-disabled build checks
that RDMA GET, PUT, and quiet fail with `RDMA is not enabled in this build`;
an RDMA-enabled build runs the RDMA lowering tests instead. UDMA lowering,
flag reservation, and diagnostic tests run in both configurations.

Pytest covers communication builders and parameters, compiler/launch ABI, bitcode
selection, SHMEM lifecycle and Matmul-ReduceScatter frontend IR, buffer geometry,
and argument validation. Lit/FileCheck covers remote-copy lowering, atomic-add
reset, dynamic GM normalization, operand/region diagnostics and the complete
example through the compiler pipeline. Native tests skip when `catlass._mlir.ir` is
unavailable; CPU-only results do not establish native compilation or NPU
correctness. Communication device correctness is covered by
[`tests/end_to_end`](tests/end_to_end/README.md), which runs real kernels on
two or more ranks on Ascend 950PR&950DT products:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  tests/end_to_end/test_remote_copy_mte.py
```

These tests cover MTE GET/PUT, GM↔UB routes, atomic ADD, padded/tail tiles,
dynamic descriptors, scratch reuse and device barrier visibility.
Pytest collection excludes these standalone scripts. All 27 MTE cases run;
atomic PUT requires P2P DMA_ADD support between the selected devices. The
distributed example additionally checks the fused Matmul-ReduceScatter result.

CI can build and run both suites from the repository root with
`bash .gitcode/scripts/pre_smoke.sh` and `pr_filelist.txt`. Set `ut_type=ir|lit|all`
to select suites, or `CI_DRY_RUN=1` to print commands without running them.
Pre-smoke builds against the repository's pinned SHMEM submodule, overriding
`TLA_DIST_DSL_SHMEM_ROOT` for this build, without configuring or building SHMEM.
Missing SHMEM configuration files disable RDMA in the DSL build. If the DSL
build fails, pre-smoke stops before running the dependent tests.
