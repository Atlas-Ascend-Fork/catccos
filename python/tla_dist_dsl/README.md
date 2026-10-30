# CATCCOS TLA_DIST DSL — Matmul-ReduceScatter

This minimal extension provides one end-to-end Ascend 950 example:
`examples/fusion_ops/matmul_reduce_scatter.py`. For rank `r`, its output is
`sum(A_p @ B_p for p in ranks)[r * (M / world_size):(r + 1) * (M / world_size), :]`.
Inputs use matching FP16 or BF16 dtypes and row-major or column-major layouts;
output and communication use FP32. M must be divisible by the rank count.

Compute operations and types come from the pinned `catlass.tla` package.
CATCCOS owns the `tla_dist` MLIR dialect, communication lowering, SHMEM device
bitcode, compiler/launch adapter, and SHMEM host lifecycle. The communication
surface consists of:

- `remote_copy(dst, src, params=None, *, peer, ub_tensor=None, flag_value=None)`:
  asynchronously copy matching RowMajor FP32 tensors. GET resolves `src` on
  the peer; PUT resolves `dst` on the peer. Source and destination column
  strides must be provably 1; row padding, dynamic shapes and dynamic row
  strides are supported. Dynamic column strides that cannot be proven to be
  1 are rejected during device lowering. MTE supports GM→GM and GM↔UB;
  GM→GM requires an explicit `CopyParams(mode="get")` or `CopyParams(mode="put")`.
  For GM↔UB, `mode` may be omitted (`None`): GM→UB selects GET and UB→GM
  selects PUT. Explicit GET/PUT values that conflict with a GM↔UB route raise
  an error. Atomic ADD requires MTE and a GM destination.
  `CopyParams` adds `mode`, `transport`, `atomic_mode`, and the
  UDMA-only `wqe_pipe` to upstream compute parameters.
- `barrier_all()`: synchronize participating ranks and AIVs.

RDMA, UDMA and SDMA support frontend IR generation. Their device compilation
explicitly reports `device lowering is not implemented`.

| Route / submission | UB scratch | Explicit flag direction |
| --- | --- | --- |
| MTE GM→GM | FP32 staging; rows padded to 32 B | MTE2→MTE3 |
| MTE GM↔UB | None | None |
| RDMA GM→GM | Control workspace, more than 64 B | SCALAR→MTE3 |
| SDMA GM→GM | At least 64 B, aligned to 64 B | MTE3→MTE2 |
| UDMA GM→GM, `wqe_pipe="mte3"` (default) | 256 B WQE workspace | MTE3→SCALAR |
| UDMA GM→GM, `wqe_pipe="s"` | None | None |

`tla.flag(name, src_pipe, dst_pipe, id=N)` creates a synchronization handle
specifying the pipe direction and hardware event id. Pass it to `remote_copy`
through `flag_value`. For MTE GM→GM copies, the helper uses this flag internally
to synchronize MTE2→MTE3 staging through UB, ensuring the load into UB completes
before the store from UB proceeds. Creating a flag does not perform
synchronization, and the internal staging synchronization does not guarantee
completion of the final store.

Scratch and flag must either both be supplied or both be absent. Control
workspaces may use UInt8. The caller guarantees scratch capacity, alignment
and lifetime; these are not checked automatically. Supplied flags must have
an explicit id in `[0, 8)` and the matching pipe direction.
The compiler runs flag lowering before communication lowering. It preserves
flag definitions with remaining consumers and writes resolved ids back to
them, so communication lowering reads ids without allocating them itself.

All communication operations require a `tla.vector()` region. For example:

```python
copy_flag = tla.flag("copy_submit", tla.arch.MTE2, tla.arch.MTE3, id=0)
copy_done = tla.flag("copy_done", tla.arch.MTE3, tla.arch.MTE2)
with tla.vector():
    tla_dist.remote_copy(
        dst, src,
        tla_dist.params.CopyParams(mode="get", transport="mte",
                                   atomic_mode=tla_dist.params.AtomicMode.ADD),
        ub_tensor=scratch, flag_value=copy_flag, peer=peer,
    )
    tla.set_flag(copy_done)
    tla.wait_flag(copy_done)
```

Returning from `remote_copy` does not imply completion. The caller arranges
completion waits, scratch reuse, remote readiness and cross-rank visibility.
The MTE GM→GM helper only synchronizes its internal MTE2→MTE3 staging edge;
it does not allocate another event for the reverse direction or wait for
the final store. For an 8×13 FP32 tile, staging needs 8×16×4 = 512 B.
RDMA/UDMA/SDMA require SHMEM quiet support when their device paths are
implemented; the DSL does not yet expose quiet.

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
```

Use the Catlass and SHMEM revisions pinned by this repository, and the
AscendNPU-IR revision pinned by Catlass. Any prebuilt AscendNPU-IR
installation must match that pinned revision.

SHMEM device headers, Python package and runtime library must use the same
revision; build/install the pinned SHMEM package using its own instructions.
NPU execution also requires compatible PyTorch and `torch_npu` installations.

When building AscendNPU-IR from source, apply the pinned Catlass patches once
before configuring it. From the CATCCOS repository root:

```bash
export CATLASS_DSL_ASCENDNPU_IR_ROOT=/path/to/AscendNPU-IR
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
export CATLASS_DSL_PREBUILT_ASCENDNPU_IR=/path/to/AscendNPU-IR
bash build.sh
python -m pip install -e .
```

`build.sh` selects pybind11 from the active Python package for both CMake
builds, avoiding a conflicting Conda prefix installation. It first
builds/installs Catlass's bundled `catlass._mlir` and `_tla_type_bridge_native.so`, then
builds `TlaDistCompile` and the communication bitcode. Compute templates are
generated with the pinned Catlass's `bc_compile` helpers during the CMake build. CATCCOS does not load a
second native MLIR bridge into Python; it exchanges MLIR text with the compiler
subprocess. The extension uses private interfaces of the pinned Catlass DSL.

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
override is forwarded to Catlass's source/install settings; these can also be
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
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python -m pytest tests -q
cmake --build csrc/mlir/build --target check-tla-dist-lit
```

Pytest covers communication builders and parameters, compiler/launch ABI, bitcode
selection, SHMEM lifecycle and Matmul-ReduceScatter frontend IR, buffer geometry,
and argument validation. Lit/FileCheck covers remote-copy lowering, atomic-add
reset, dynamic GM normalization, operand/region diagnostics and the complete
example through the compiler pipeline. Native tests skip when `catlass._mlir.ir` is
unavailable; CPU-only results do not establish native compilation or NPU
correctness. Communication device correctness is covered by
[`tests/end_to_end`](tests/end_to_end/README.md), which runs real kernels on
two or more Ascend 950 ranks:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  -m pytest tests/end_to_end -q -ra
```

These tests cover MTE GET/PUT, GM↔UB routes, atomic ADD, padded/tail tiles,
dynamic descriptors, scratch reuse and device barrier visibility. Ordinary
single-process pytest skips them. The distributed example additionally checks
the fused Matmul-ReduceScatter result. Four atomic PUT cases are explicitly
skipped because the current Ascend 950 runtime hangs during their execution;
see the end-to-end test README for details.

CI can build and run both suites from the repository root with
`bash .gitcode/scripts/pre_smoke.sh` and `pr_filelist.txt`. Set `ut_type=ir|lit|all`
to select suites, or `CI_DRY_RUN=1` to print commands without running them.
