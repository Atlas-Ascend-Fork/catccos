# Communication API end-to-end tests

These tests compile CATCCOS DSL kernels, launch them on real Ascend 950 NPUs,
and compare every rank's FP32 output with an exact CPU reference. No compiler,
device operation or SHMEM call is mocked.

From `python/tla_dist_dsl`, after building/installing the DSL and the pinned
SHMEM package and activating the CANN/PyTorch/torch_npu environment:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
ASCEND_RT_VISIBLE_DEVICES=0,1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
timeout --signal=TERM --kill-after=30s 180s \
torchrun --standalone --nproc-per-node=2 -m pytest tests/end_to_end -q -ra
```

Select any two available devices with `ASCEND_RT_VISIBLE_DEVICES`. More local
ranks are supported by increasing both the visible device list and
`--nproc-per-node`. All ranks must execute the same test selection in the same
order; run without pytest-xdist. The Gloo control process group has a 180-second
timeout. The external timeout above also limits device-level hangs; adjust it
for slower builds or larger rank counts.

Coverage:

- MTE GM→GM GET and PUT overwrite, and GET with atomic ADD.
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
Ascend 950 hardware are unavailable. Compilation and numerical errors fail all
ranks together. Each rank uses its own temporary compilation cache, and SHMEM
allocations are freed before the process group is destroyed.

Kernels launch one AIC and its two paired AIVs, selecting the mixed SHMEM
hardware core barrier template used by the fused example. Only sub-block 0
copies data; both AIVs join every device barrier.

Four cross-rank atomic PUT cases are explicitly skipped: three copy geometries
and the PUT variant of the atomic-state reset test. The current PCIe PIX
topology reports no P2P DMA_ADD capability and hangs during device synchronization
for remote atomic writes, reproduced with both DSL and native Ascend C kernels.
Remove these skip marks only when testing a device pair with verified support.

RDMA, UDMA and SDMA are excluded because their device lowering is not yet
implemented; their frontend contracts remain covered in `tests/frontend/ops/comm`.
