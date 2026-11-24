# Communication API end-to-end tests

These tests compile CATCCOS DSL kernels, launch them on NPUs in Ascend 950PR&950DT products,
and compare every rank's FP32 output with an exact CPU reference. No compiler,
device operation or SHMEM call is mocked.

`test_remote_copy_mte.py`, `test_remote_copy_udma.py`, and
`test_remote_copy_rdma.py` are standalone torchrun scripts. Each contains its own
kernels, case enumeration, initialization, comparisons, and cleanup; none imports
pytest or another test file.

Note: the MTE and UDMA example(i.e. `test_remote_copy_mte.py` and
`test_remote_copy_udma.py`) need UB(Unified Bus) link between devices, can
execute `npu-smi info -t topo` check hardware communication link between devices.

From `python/tla_dist_dsl`, after building/installing the DSL and the pinned
SHMEM package and activating the CANN/PyTorch/torch_npu environment:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
ASCEND_RT_VISIBLE_DEVICES=0,1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
timeout --signal=TERM --kill-after=30s 180s \
torchrun --standalone --nproc-per-node=2 \
  tests/end_to_end/test_remote_copy_mte.py
```

Select any two available devices with `ASCEND_RT_VISIBLE_DEVICES`. More local
ranks are supported by increasing both the visible device list and
`--nproc-per-node`. All ranks must execute the same test selection in the same
order. The Gloo control process group has a 180-second timeout.
The external timeout above also limits device-level hangs; adjust it
for slower builds or larger rank counts.

Coverage:

- MTE GM→GM GET and PUT, both overwrite and atomic ADD.
- GM→UB GET and UB→GM PUT, explicit and inferred modes.
- Aligned tiles, tail columns, distinct padded source/destination row strides,
  static descriptors and dynamic shape/stride arguments.
- Repeated copies with explicit completion waits before UB reuse, and ordinary
  stores following atomic ADD to verify that atomic state is reset.
- Device `barrier_all()` visibility: a ring PUT is read back with GET inside
  the same kernel over three rounds, with two participating AIVs per rank.
- Source preservation and destination padding guards.

Ordinary single-process pytest runs skip this directory before importing NPU
dependencies. Distributed runs skip if native bindings, SHMEM, the compiler or
hardware from Ascend 950PR&950DT products are unavailable. Compilation and numerical errors fail all
ranks together. Each rank uses its own temporary compilation cache, and SHMEM
allocations are freed before the process group is destroyed.

MTE kernels launch one AIC and its two paired AIVs, selecting the mixed SHMEM
hardware core barrier template used by the fused example. Only sub-block 0
copies data; both AIVs join every device barrier.

MTE atomic PUT tests require P2P DMA_ADD support between the selected devices.
All 27 MTE cases run, including three atomic PUT copy geometries and the PUT
variant of the atomic-state reset test.

Build CATCCOS with `build.sh` and install the SHMEM Python package from the
submodule. For RDMA, first build SHMEM with RDMA enabled and its XSCALE or
HNS_1825 backend selected. CATCCOS reads that configuration automatically,
as described in the [build instructions](../../README.md#dependencies-and-build).
Engine selection and Host barriers use SHMEM's public Python bindings, with
one transport per process group. From `python/tla_dist_dsl`, using the
environment described above:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
ASCEND_RT_VISIBLE_DEVICES=0,1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
timeout --signal=TERM --kill-after=30s 420s \
torchrun --standalone --nproc-per-node=2 \
  tests/end_to_end/test_remote_copy_rdma.py

PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
ASCEND_RT_VISIBLE_DEVICES=0,1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
timeout --signal=TERM --kill-after=30s 420s \
torchrun --standalone --nproc-per-node=2 \
  tests/end_to_end/test_remote_copy_udma.py
```

RDMA needs configured interfaces on every selected device; UDMA needs device
UDMA support. Two ranks cover all current cases. Initialization failures fail the run.

- Case counts: MTE 27, RDMA 14, UDMA 23.
- RDMA/UDMA cover GET/PUT, AIV and mixed-AIV, static/dynamic layouts, padded
  tails, GM storage offsets and UB tile offsets; UDMA covers both WQE pipes.
- Ordinary cases use a zero-offset UB scratch: 128 bytes for RDMA and 256 bytes
  for UDMA, both 32-byte aligned. Each transport has one `boundary/ub-offset`
  case that passes the second row of a larger UB allocation.
- Batch cases submit two copies per quiet for three rounds and read both results
  inside the kernel. PUT receivers read after all senders quiet and join a rank barrier.
- Other cases cover mixed GET/PUT, GET-to-PUT dependencies, empty copies and
  repeated quiet. All cases compare against CPU references and check padding and sources.

Use `--list` to list cases without initializing SHMEM or an NPU, and
`--case SUBSTRING` to filter them. These options require the DSL package and native bindings.

```bash
python tests/end_to_end/test_remote_copy_mte.py --list
python tests/end_to_end/test_remote_copy_rdma.py --list
python tests/end_to_end/test_remote_copy_udma.py --list --case basic/s/put
# With the same NPU environment as above:
torchrun --standalone --nproc-per-node=2 \
  tests/end_to_end/test_remote_copy_mte.py --case gm-to-gm/copy
torchrun --standalone --nproc-per-node=2 \
  tests/end_to_end/test_remote_copy_udma.py --case batch-one-quiet
```

Rank 0 prints RUN/PASS/FAIL and final counts.
The first failed case stops the run with a nonzero status; resources are cleaned up.

Large contiguous segments are split into NBI requests: RDMA uses at most
`UINT32_MAX / sizeof(float)` elements per request (4 GiB minus 4 B), and UDMA
uses at most 256 MiB. Contiguous tiles are merged before splitting; padded
tiles are split within each row. The caller still issues quiet for completion.
SDMA device lowering is unimplemented.

Integer `event_id` validation:
all 27 MTE and 23 UDMA cases passed. The MTE double-buffered Matmul-ReduceScatter
example also passed on both ranks with its default dimensions.
All 14 RDMA kernels lowered and linked for XSCALE and both HNS_1825 PI modes
(42 device binaries). RDMA device execution remains pending: the installed
SHMEM lacks RDMA v2 support, so `aclshmemx_set_qp_num` returned
`ACLSHMEM_NOT_SUPPORTED` (`-100`) before initialization.

HNS_1825 submission-side CQ polling requires a separate hardware acceptance run.
The existing batch case submits 16 requests before each quiet, which does not
guarantee a full queue. With `D = min(SQ.depth, CQ.depth)` from the actual SHMEM
queue context, reuse `quiet_observe` with `MultiSubmitRemoteCase.count` set to
at least `ceil((D - 9) / 8)`: each padded 8-row tile produces eight requests,
and HNS polls when `head - tail + 10 >= D`. Then quiet and compare all results.
This keeps the default case count unchanged.

RDMA/UDMA tests use `ShmemRuntime(..., qp_num=2)`. The `multi-qp/` cases
run two AIVs against the same peer, using sub-block indices 0 and 1 as their
QPs and checking both destination tiles. These cover GET/PUT, including both
UDMA WQE pipes; other cases use the default QP 0.
Submission and quiet share `CopyParams` in the multi-QP kernels.

Frontend tests check that MTE ignores both nonzero constant and runtime QP values.
