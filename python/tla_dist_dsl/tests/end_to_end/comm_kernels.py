"""Small kernels isolating the supported communication routes and ordering."""

from dataclasses import dataclass

from catccos import tla_dist  # isort: skip - bootstrap pinned Catlass before importing it
from catlass import tla


@dataclass(frozen=True)
class CopyCase:
    rows: int
    columns: int
    route: str = "gm_to_gm"
    mode: str | None = None
    atomic: bool = False
    reset_atomic: bool = False
    repeats: int = 2


@tla.kernel
def remote_copy(dst: tla.Tensor, src: tla.Tensor, peer: tla.Int32, case: tla.Constexpr[CopyCase]):
    # Bind compact physical buffers on the host and crop logical columns here,
    # retaining each buffer's distinct physical row stride in the tile descriptor.
    tile_shape = tla.make_shape(case.rows, case.columns)
    src = tla.tile_view(src, tile_shape, tla.make_coord(0, 0))
    dst = tla.tile_view(dst, tile_shape, tla.make_coord(0, 0))
    # Each FP32 UB row occupies whole 32-byte blocks, including tail columns.
    padded_columns = ((case.columns + 7) // 8) * 8
    scratch_ptr = tla.allocate(case.rows * padded_columns, tla.Float32, tla.AddressSpace.ub, 32)
    scratch = tla.make_tensor(
        scratch_ptr,
        tla.make_layout(
            tla.make_shape(case.rows, padded_columns),
            tla.make_stride(padded_columns, 1),
            origin_shape=tla.make_shape(case.rows, case.columns),
            layoutTag=tla.arch.RowMajor,
        ),
    )
    submit = tla.flag("copy_submit", tla.arch.MTE2, tla.arch.MTE3, id=0)
    complete = tla.flag("copy_complete", tla.arch.MTE3, tla.arch.MTE2, id=0)
    params = tla_dist.params.CopyParams(
        mode=case.mode,
        atomic_mode=tla_dist.params.AtomicMode.ADD if case.atomic else tla_dist.params.AtomicMode.NONE,
    )
    # Select the same mixed-kernel hardware barrier path as the fused example.
    with tla.cube():
        tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        tla_dist.barrier_all()
        if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
            for _ in tla.range(case.repeats):
                if tla.const_expr(case.route == "gm_to_gm"):
                    tla_dist.remote_copy(
                        dst,
                        src,
                        params,
                        peer=peer,
                        ub_tensor=scratch,
                        flag_value=submit,
                    )
                elif tla.const_expr(case.route == "gm_to_ub"):
                    tla_dist.remote_copy(scratch, src, params, peer=peer)
                    tla.set_flag(submit)
                    tla.wait_flag(submit)
                    tla.copy(dst, scratch)
                else:
                    tla.copy(scratch, src)
                    tla.set_flag(submit)
                    tla.wait_flag(submit)
                    tla_dist.remote_copy(dst, scratch, params, peer=peer)
                # Wait for the final store before reusing this UB slot.
                tla.set_flag(complete)
                tla.wait_flag(complete)

            if tla.const_expr(case.reset_atomic):
                tla_dist.remote_copy(
                    dst,
                    src,
                    tla_dist.params.CopyParams(mode=case.mode),
                    peer=peer,
                    ub_tensor=scratch,
                    flag_value=submit,
                )
                tla.set_flag(complete)
                tla.wait_flag(complete)
        # Idle AIVs also participate; completion waits precede rank visibility.
        tla_dist.barrier_all()


@tla.kernel
def barrier_visibility(
    dst: tla.Tensor,
    src: tla.Tensor,
    observed: tla.Tensor,
    peer: tla.Int32,
    case: tla.Constexpr[CopyCase],
):
    padded_columns = ((case.columns + 7) // 8) * 8
    scratch_ptr = tla.allocate(case.rows * padded_columns, tla.Float32, tla.AddressSpace.ub, 32)
    scratch = tla.make_tensor(
        scratch_ptr,
        tla.make_layout(
            tla.make_shape(case.rows, padded_columns),
            tla.make_stride(padded_columns, 1),
            origin_shape=tla.make_shape(case.rows, case.columns),
            layoutTag=tla.arch.RowMajor,
        ),
    )
    submit = tla.flag("barrier_submit", tla.arch.MTE2, tla.arch.MTE3, id=0)
    complete = tla.flag("barrier_complete", tla.arch.MTE3, tla.arch.MTE2, id=0)
    tile_shape = tla.make_shape(case.rows, case.columns)
    with tla.cube():
        tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        for iteration in tla.range(case.repeats):
            source_tile = tla.tile_view(src, tile_shape, tla.make_coord(iteration, 0))
            observed_tile = tla.tile_view(observed, tile_shape, tla.make_coord(iteration, 0))
            if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
                tla_dist.remote_copy(
                    dst,
                    source_tile,
                    tla_dist.params.CopyParams(mode="put"),
                    peer=peer,
                    ub_tensor=scratch,
                    flag_value=submit,
                )
                tla.set_flag(complete)
                tla.wait_flag(complete)
            # Read back the next rank's destination only after all writers finish.
            tla_dist.barrier_all()
            if tla.arch.block_idx() == 0 and tla.arch.sub_block_idx() == 0:
                tla_dist.remote_copy(
                    observed_tile,
                    dst,
                    tla_dist.params.CopyParams(mode="get"),
                    peer=peer,
                    ub_tensor=scratch,
                    flag_value=submit,
                )
                tla.set_flag(complete)
                tla.wait_flag(complete)
            # Readers must finish before the next iteration overwrites dst.
            tla_dist.barrier_all()
