# Reviewed Catlass source patch

The pinned Catlass revision `76946bb3feb6efef4efb204a74fff44196b73c14`
already represents UInt64 as the Python descriptor token `u64` and native MLIR
type `ui64`, but its `make_tensor` and `make_tensor_like` constructors reject
that dtype. CATCCOS MTE communication needs an unsigned 64-bit UB scratch view.

`catlass/uint64-ub-tensor.patch` adds only that constructor case:
`dtype == "u64" and ptr_ty.addrspace == "ub"`. It does not change Catlass's
global dtype whitelist, signed Int64, other address spaces, arithmetic, or
local-copy capability. CATCCOS keeps its original Catlass gitlink.

The build applies this source patch before invoking Catlass's build. It is
separate from Catlass's existing AscendNPU-IR visibility patch. The public
`tla.make_tensor` / `tla.make_tensor_like` APIs then construct the UInt64 UB
tensor; no runtime monkeypatch or substitute payload dtype is used.

From the CATCCOS repository root:

```bash
python python/tla_dist_dsl/tools/patches/apply_catlass_patches.py \
  --catlass-root 3rdparty/catlass --check
python python/tla_dist_dsl/tools/patches/apply_catlass_patches.py \
  --catlass-root 3rdparty/catlass
```

`--check` is read-only: it emits `needs_apply` for the reviewed original source
and `already_applied` for the exact patched source. Default mode applies the
patch once (`applied`) and accepts an exact repeat (`already_applied`). Both
modes reject revision drift, altered patch bytes, unexpected target files,
source drift, or unrelated tracked changes. Nested submodule working-tree
changes are ignored by this check because AscendNPU-IR has its own reviewed
visibility patch; nested submodule commit changes remain detectable.

`catlass-uint64-ub.json` records the pinned revision, target path, original and
patched source SHA256, and patch SHA256. Before changing the real tree, the
helper applies the patch to a temporary copy and verifies its result. Its
successful stdout is one JSON object with `status`, `catlass_revision`,
`patch_sha256`, `source_sha256`, `expected_sha256`, `preimage_sha256`, `target`,
and `tracked_changes`. Deployment receipts must require `applied` or
`already_applied` and `source_sha256 == expected_sha256`; `needs_apply` does
not satisfy the dependency gate. Errors exit nonzero with a JSON diagnostic.

The dependency identity is the pinned revision **plus** this patch's hash and
the verified source hash. A dirty submodule alone is insufficient evidence.
The dedicated patch tests use disposable sparse clones. Native frontend tests
verify unsigned UB construction and the forbidden signed/address-space/local
copy combinations; the E2E suite still establishes actual communication.
