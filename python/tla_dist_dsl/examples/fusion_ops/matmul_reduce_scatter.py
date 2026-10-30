"""Ascend 950 Matmul-ReduceScatter implemented with the TLA DSL.

For every rank ``r`` this example computes ``C_r = A_r @ B_r``.  The rows of
``C_r`` are partitioned by dst rank.  In each communication round all launched
AICs distribute the MMAD blocks with a grid-stride loop.  Blocks for the local
destination are written directly to the ordinary-GM output shard; blocks for
other destinations are published in ACLSHMEM symmetric GM.  After each AIC
signals both of its paired AIVs and all ranks synchronize, every AIV pulls a
disjoint subset of the local destination slots from remote ranks and atomically
adds them to the output shard.

The communication batch scales with the runtime AIC count, following the
CatCCOS ``matmul_reduce_scatter.hpp`` scheduler.  Two symmetric-workspace
stages let AIC produce round ``i + 1`` while AIV consumes round ``i``.  A
stage-specific reverse acknowledgement prevents AIC from reusing a stage
until all ranks have finished reading it.

``TILING`` controls L1/L0 geometry, communication block/tile shapes, the
communication interval and the AIC core split. Only sub-block 0 of each selected
logical AIV core performs data transfers; all AIVs still join global barriers.
``SWIZZLE`` configures the MMAD serpentine traversal and the communication
block/peer grouping with rank shift. Deterministic communication assigns all
peer contributions for a block to the same AIV in a fixed order. Its offset
counts groups of the selected AIVs; otherwise it scales the core split along
the selected communication direction.

L1 and L0A/L0B use ping-pong buffers following ``matmul_add_ub.py``; L0C
retains a single accumulator protected by FIX/CUBE flags. Communication UB
also alternates slots, waiting for each slot's previous MTE3 store before
reusing it for an MTE2 load. L1, L0A/L0B, communication UB and symmetric
workspace all use a fixed depth of two stages.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Import TLA_DIST first so it bootstraps the pinned Catlass package and unified
# native type bridge before the compute DSL is imported directly.
from catccos import tla_dist
from catlass import tla
from catlass import types as tla_types

# Support direct execution from the fusion_ops directory.
sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import SwizzleParams, TilingParams

TILING = TilingParams()
SWIZZLE = SwizzleParams()

WORKSPACE_STAGES = 2

CONTROL_PLANE_BACKEND = "gloo"
SHMEM_UID_BUFFER_SIZE = 512
EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = EXAMPLE_DIR / "artifacts" / "runtime-cache"


@tla.jit
def mmad_swizzle(
    block_idx: tla.Int32,
    block_rows: tla.Int32,
    block_cols: tla.Int32,
    _swizzle: SwizzleParams,
) -> tuple[tla.Int32, tla.Int32]:
    """Map a linear MMAD block to grouped serpentine matrix coordinates.

    Shared by cube production and vector consumption. Workspace slots retain
    their linear indices; only matrix coordinates follow this traversal.
    """
    if tla.const_expr(_swizzle.MMAD_SWIZZLE_DIRECTION == 0):
        group_size = _swizzle.MMAD_SWIZZLE_OFFSET * block_cols
        group_idx = block_idx // group_size
        group_offset = block_idx % group_size
        group_rows = min(
            _swizzle.MMAD_SWIZZLE_OFFSET,
            block_rows - group_idx * _swizzle.MMAD_SWIZZLE_OFFSET,
        )
        block_row = group_idx * _swizzle.MMAD_SWIZZLE_OFFSET + group_offset % group_rows
        block_col = group_offset // group_rows
        if group_idx % 2 == 1:
            block_col = block_cols - block_col - 1
    else:
        group_size = _swizzle.MMAD_SWIZZLE_OFFSET * block_rows
        group_idx = block_idx // group_size
        group_offset = block_idx % group_size
        group_cols = min(
            _swizzle.MMAD_SWIZZLE_OFFSET,
            block_cols - group_idx * _swizzle.MMAD_SWIZZLE_OFFSET,
        )
        block_col = group_idx * _swizzle.MMAD_SWIZZLE_OFFSET + group_offset % group_cols
        block_row = group_offset // group_cols
        if group_idx % 2 == 1:
            block_row = block_rows - block_row - 1
    return block_row, block_col


@tla.jit
def comm_swizzle(
    comm_task: tla.Int32,
    comm_blocks_this_rank: tla.Int32,
    world_size: tla.Int32,
    comm_offset: tla.Int32,
    peer_stride: tla.Int32,
    _swizzle: SwizzleParams,
) -> tuple[tla.Int32, tla.Int32]:
    """Map a communication task to a block index and rank-shifted peer.

    The caller supplies the effective group offset and peer stride, and pads
    the task count in deterministic mode. Returned block indices may lie in
    that padding; the caller must skip them and contributions from itself.
    """
    if tla.const_expr(_swizzle.COMM_SWIZZLE_DIRECTION == 0):
        comm_group_size = comm_offset * world_size
        comm_group = comm_task // comm_group_size
        comm_in_group = comm_task % comm_group_size
        comm_group_rows = comm_offset
        if tla.const_expr(not _swizzle.IS_DETERMINISTIC):
            comm_group_rows = min(comm_offset, comm_blocks_this_rank - comm_group * comm_offset)
        comm_block_idx = comm_group * comm_offset + comm_in_group % comm_group_rows
        peer_coord = comm_in_group // comm_group_rows
    else:
        comm_group_size = comm_offset * comm_blocks_this_rank
        comm_group = comm_task // comm_group_size
        comm_in_group = comm_task % comm_group_size
        comm_group_cols = min(comm_offset, world_size - comm_group * comm_offset)
        comm_block_idx = comm_in_group // comm_group_cols
        peer_coord = comm_group * comm_offset + comm_in_group % comm_group_cols

    peer_offset = peer_coord * peer_stride
    peer = (peer_offset + peer_offset // world_size + comm_block_idx) % world_size
    return comm_block_idx, peer


@dataclass(frozen=True)
class _MmadBuffers:
    """L1/L0A/L0B ping-pong storage and a single L0C accumulator per AIC."""

    l1a_ptr_0: tla.Pointer
    l1a_ptr_1: tla.Pointer
    l1b_ptr_0: tla.Pointer
    l1b_ptr_1: tla.Pointer
    l0a_ptr_0: tla.Pointer
    l0a_ptr_1: tla.Pointer
    l0b_ptr_0: tla.Pointer
    l0b_ptr_1: tla.Pointer
    l0c_ptr: tla.Pointer


@dataclass(frozen=True)
class _MmadFlags:
    """Pipeline events whose lifetime spans all MMAD blocks."""

    l1a_loaded_0: tla_types.TlaFlag
    l1a_loaded_1: tla_types.TlaFlag
    l1b_loaded_0: tla_types.TlaFlag
    l1b_loaded_1: tla_types.TlaFlag
    l1a_available_0: tla_types.TlaFlag
    l1a_available_1: tla_types.TlaFlag
    l1b_available_0: tla_types.TlaFlag
    l1b_available_1: tla_types.TlaFlag
    l0a_available_0: tla_types.TlaFlag
    l0a_available_1: tla_types.TlaFlag
    l0b_available_0: tla_types.TlaFlag
    l0b_available_1: tla_types.TlaFlag
    l0_ab_loaded: tla_types.TlaFlag
    l0c_ready: tla_types.TlaFlag
    l0c_available: tla_types.TlaFlag


@tla.jit
def block_mmad(
    a_block: tla.Tensor,
    b_block: tla.Tensor,
    c_block: tla.Tensor,
    buffers: _MmadBuffers,
    flags: _MmadFlags,
    l1_buf_idx: tla.Int32,
    l0_buf_idx: tla.Int32,
    _tiling: TilingParams,
) -> tuple[tla.Int32, tla.Int32]:
    """Compute and store one GM block, returning the next L1/L0 slots.

    Call inside a cube region with initialized availability flags. Store L0C
    into c_block, then signal its availability on FIX. Keep the returned slot
    indices across MN blocks and communication rounds.
    """
    l0_c = tla.make_tensor_like(buffers.l0c_ptr, c_block, tla.arch.L0Clayout)
    k = a_block.origin_shape[1]
    l1_k_loops = (k + _tiling.l1_tk - 1) // _tiling.l1_tk

    # Do not overwrite L0C until the previous MMAD block has
    # completely left that storage level.  K is processed in
    # _tiling.l1_tk chunks, with a dynamically cropped final chunk.
    tla.wait_flag(flags.l0c_available)

    for l1_k_idx in tla.range(0, l1_k_loops, 1):
        a_l1_block = tla.tile_view(
            a_block,
            tla.make_shape(_tiling.l1_tm, _tiling.l1_tk),
            tla.make_coord(0, l1_k_idx),
        )
        b_l1_block = tla.tile_view(
            b_block,
            tla.make_shape(_tiling.l1_tk, _tiling.l1_tn),
            tla.make_coord(l1_k_idx, 0),
        )
        l1_a = tla.make_tensor_like(buffers.l1a_ptr_0 if l1_buf_idx == 0 else buffers.l1a_ptr_1, a_l1_block)
        l1_b = tla.make_tensor_like(buffers.l1b_ptr_0 if l1_buf_idx == 0 else buffers.l1b_ptr_1, b_l1_block)

        if l1_buf_idx == 0:
            tla.wait_flag(flags.l1a_available_0)
        else:
            tla.wait_flag(flags.l1a_available_1)
        tla.copy(l1_a, a_l1_block)
        if l1_buf_idx == 0:
            tla.set_flag(flags.l1a_loaded_0)
        else:
            tla.set_flag(flags.l1a_loaded_1)
        if l1_buf_idx == 0:
            tla.wait_flag(flags.l1b_available_0)
        else:
            tla.wait_flag(flags.l1b_available_1)
        tla.copy(l1_b, b_l1_block)
        if l1_buf_idx == 0:
            tla.set_flag(flags.l1b_loaded_0)
        else:
            tla.set_flag(flags.l1b_loaded_1)

        l0_k_loops = (l1_a.origin_shape[1] + _tiling.l0_tk - 1) // _tiling.l0_tk

        for l0_k_idx in tla.range(0, l0_k_loops, 1):
            l1_a_tile = tla.tile_view(
                l1_a,
                tla.make_shape(_tiling.l0_tm, _tiling.l0_tk),
                tla.make_coord(0, l0_k_idx),
            )
            l1_b_tile = tla.tile_view(
                l1_b,
                tla.make_shape(_tiling.l0_tk, _tiling.l0_tn),
                tla.make_coord(l0_k_idx, 0),
            )
            l0_a = tla.make_tensor_like(
                buffers.l0a_ptr_0 if l0_buf_idx == 0 else buffers.l0a_ptr_1,
                l1_a_tile,
                tla.arch.zN,
            )
            l0_b = tla.make_tensor_like(
                buffers.l0b_ptr_0 if l0_buf_idx == 0 else buffers.l0b_ptr_1,
                l1_b_tile,
                tla.arch.nZ,
            )

            if l0_k_idx == 0:
                if l1_buf_idx == 0:
                    tla.wait_flag(flags.l1a_loaded_0)
                else:
                    tla.wait_flag(flags.l1a_loaded_1)
            if l0_buf_idx == 0:
                tla.wait_flag(flags.l0a_available_0)
            else:
                tla.wait_flag(flags.l0a_available_1)
            tla.copy(l0_a, l1_a_tile)
            if l0_k_idx == l0_k_loops - 1:
                if l1_buf_idx == 0:
                    tla.set_flag(flags.l1a_available_0)
                else:
                    tla.set_flag(flags.l1a_available_1)

            if l0_k_idx == 0:
                if l1_buf_idx == 0:
                    tla.wait_flag(flags.l1b_loaded_0)
                else:
                    tla.wait_flag(flags.l1b_loaded_1)
            if l0_buf_idx == 0:
                tla.wait_flag(flags.l0b_available_0)
            else:
                tla.wait_flag(flags.l0b_available_1)
            tla.copy(l0_b, l1_b_tile)
            if l0_k_idx == l0_k_loops - 1:
                if l1_buf_idx == 0:
                    tla.set_flag(flags.l1b_available_0)
                else:
                    tla.set_flag(flags.l1b_available_1)

            tla.set_flag(flags.l0_ab_loaded)
            tla.wait_flag(flags.l0_ab_loaded)
            tla.mmad(
                l0_c,
                l0_a,
                l0_b,
                init_c=(l1_k_idx == 0 and l0_k_idx == 0),
            )
            if l0_buf_idx == 0:
                tla.set_flag(flags.l0a_available_0)
                tla.set_flag(flags.l0b_available_0)
            else:
                tla.set_flag(flags.l0a_available_1)
                tla.set_flag(flags.l0b_available_1)

            l0_buf_idx = 1 - l0_buf_idx
        l1_buf_idx = 1 - l1_buf_idx

    tla.set_flag(flags.l0c_ready)
    tla.wait_flag(flags.l0c_ready)
    tla.copy(c_block, l0_c)
    tla.set_flag(flags.l0c_available)
    return l1_buf_idx, l0_buf_idx


@dataclass(frozen=True)
class _CommBuffers:
    """Ping-pong UB storage shared by communication blocks on an AIV."""

    remote_ub_ptr_0: tla.Pointer
    remote_ub_ptr_1: tla.Pointer


@dataclass(frozen=True)
class _CommFlags:
    """Submission and completion events for each communication UB slot."""

    remote_available_0: tla_types.TlaFlag
    remote_available_1: tla_types.TlaFlag
    remote_submit_0: tla_types.TlaFlag
    remote_submit_1: tla_types.TlaFlag


@tla.jit
def block_comm(
    output_block: tla.Tensor,
    workspace_block: tla.Tensor,
    peer: tla.Int32,
    buffers: _CommBuffers,
    flags: _CommFlags,
    remote_buf_idx: tla.Int32,
    _tiling: TilingParams,
) -> tla.Int32:
    """Pull one communication block from peer and atomically add it to output.

    Call inside a vector region with initialized availability flags and a
    published remote workspace. Keep the returned UB slot index across blocks
    and rounds; the caller handles workspace handshakes and rank barriers.
    """
    comm_tile_shape = tla.make_shape(_tiling.comm_tile_m, _tiling.comm_tile_n)
    tile_m_this_block = (output_block.origin_shape[0] + _tiling.comm_tile_m - 1) // _tiling.comm_tile_m
    tile_n_this_block = (output_block.origin_shape[1] + _tiling.comm_tile_n - 1) // _tiling.comm_tile_n
    tiles_this_block = tile_m_this_block * tile_n_this_block

    for tile_idx in tla.range(0, tiles_this_block, 1):
        tile_row_idx = tile_idx // tile_n_this_block
        tile_col_idx = tile_idx - tile_row_idx * tile_n_this_block

        workspace_tile = tla.tile_view(
            workspace_block,
            comm_tile_shape,
            tla.make_coord(
                tile_row_idx,
                tile_col_idx,
            ),
        )
        output_tile = tla.tile_view(
            output_block,
            comm_tile_shape,
            tla.make_coord(
                tile_row_idx,
                tile_col_idx,
            ),
        )
        # MTE2 may refill a slot only after its previous MTE3
        # atomic store completes. Each slot also owns its
        # MTE2->MTE3 submission event, allowing overlap.
        if remote_buf_idx == 0:
            tla.wait_flag(flags.remote_available_0)
            remote_ub = tla.make_tensor_like(buffers.remote_ub_ptr_0, workspace_tile, tla.arch.RowMajor)
            tla_dist.remote_copy(
                output_tile,
                workspace_tile,
                tla_dist.params.CopyParams(mode="get", atomic_mode=tla.params.AtomicMode.ADD),
                ub_tensor=remote_ub,
                flag_value=flags.remote_submit_0,
                peer=peer,
            )
            tla.set_flag(flags.remote_available_0)
        else:
            tla.wait_flag(flags.remote_available_1)
            remote_ub = tla.make_tensor_like(buffers.remote_ub_ptr_1, workspace_tile, tla.arch.RowMajor)
            tla_dist.remote_copy(
                output_tile,
                workspace_tile,
                tla_dist.params.CopyParams(mode="get", atomic_mode=tla.params.AtomicMode.ADD),
                ub_tensor=remote_ub,
                flag_value=flags.remote_submit_1,
                peer=peer,
            )
            tla.set_flag(flags.remote_available_1)
        remote_buf_idx = 1 - remote_buf_idx
    return remote_buf_idx


@tla.kernel
def matmul_reduce_scatter(
    a_gm: tla.Tensor,
    b_gm: tla.Tensor,
    symmetric_workspace: tla.Tensor,
    output_shard: tla.Tensor,
    rank: tla.Int32,
    world_size: tla.Int32,
    _tiling: TilingParams,
    _swizzle: SwizzleParams,
) -> None:
    m = a_gm.origin_shape[0]
    n = b_gm.origin_shape[1]
    k = a_gm.origin_shape[1]
    shard_m = m // world_size
    block_rows_per_shard = (shard_m + _tiling.l1_tm - 1) // _tiling.l1_tm
    block_cols_per_shard = (n + _tiling.l1_tn - 1) // _tiling.l1_tn
    blocks_per_shard = block_rows_per_shard * block_cols_per_shard

    l1a_loaded_0 = tla.flag("l1a_loaded_0", tla.arch.MTE2, tla.arch.MTE1)
    l1a_loaded_1 = tla.flag("l1a_loaded_1", tla.arch.MTE2, tla.arch.MTE1)
    l1b_loaded_0 = tla.flag("l1b_loaded_0", tla.arch.MTE2, tla.arch.MTE1)
    l1b_loaded_1 = tla.flag("l1b_loaded_1", tla.arch.MTE2, tla.arch.MTE1)
    l1a_available_0 = tla.flag("l1a_available_0", tla.arch.MTE1, tla.arch.MTE2)
    l1a_available_1 = tla.flag("l1a_available_1", tla.arch.MTE1, tla.arch.MTE2)
    l1b_available_0 = tla.flag("l1b_available_0", tla.arch.MTE1, tla.arch.MTE2)
    l1b_available_1 = tla.flag("l1b_available_1", tla.arch.MTE1, tla.arch.MTE2)
    l0a_available_0 = tla.flag("l0a_available_0", tla.arch.CUBE, tla.arch.MTE1)
    l0a_available_1 = tla.flag("l0a_available_1", tla.arch.CUBE, tla.arch.MTE1)
    l0b_available_0 = tla.flag("l0b_available_0", tla.arch.CUBE, tla.arch.MTE1)
    l0b_available_1 = tla.flag("l0b_available_1", tla.arch.CUBE, tla.arch.MTE1)
    remote_available_0 = tla.flag("remote_available_0", tla.arch.MTE3, tla.arch.MTE2)
    remote_available_1 = tla.flag("remote_available_1", tla.arch.MTE3, tla.arch.MTE2)
    remote_submit_0 = tla.flag("remote_submit_0", tla.arch.MTE2, tla.arch.MTE3, id=0)
    remote_submit_1 = tla.flag("remote_submit_1", tla.arch.MTE2, tla.arch.MTE3, id=1)
    l0_ab_loaded = tla.flag("l0_ab_loaded", tla.arch.MTE1, tla.arch.CUBE)
    l0c_ready = tla.flag("l0c_ready", tla.arch.CUBE, tla.arch.FIX)
    l0c_available = tla.flag("l0c_available", tla.arch.FIX, tla.arch.CUBE)

    # Each workspace stage owns an independent mode=2 handshake.  One AIC
    # operation therefore synchronizes with both paired AIV sub-blocks without
    # allowing back-to-back rounds to overwrite the same hardware event.
    aic_to_aiv_stage_0 = tla.cross_flag("aic_to_aiv_stage_0", mode=2)
    aic_to_aiv_stage_1 = tla.cross_flag("aic_to_aiv_stage_1", mode=2)
    aiv_to_aic_stage_0 = tla.cross_flag("aiv_to_aic_stage_0", mode=2)
    aiv_to_aic_stage_1 = tla.cross_flag("aiv_to_aic_stage_1", mode=2)

    l1a_ptr_0 = tla.allocate(_tiling.l1_tm * _tiling.l1_tk, a_gm.ptr.dtype, tla.AddressSpace.l1, 512)
    l1b_ptr_0 = tla.allocate(_tiling.l1_tk * _tiling.l1_tn, b_gm.ptr.dtype, tla.AddressSpace.l1, 512)
    l1a_ptr_1 = tla.allocate(_tiling.l1_tm * _tiling.l1_tk, a_gm.ptr.dtype, tla.AddressSpace.l1, 512)
    l1b_ptr_1 = tla.allocate(_tiling.l1_tk * _tiling.l1_tn, b_gm.ptr.dtype, tla.AddressSpace.l1, 512)
    l0a_ptr_0 = tla.allocate(_tiling.l0_tm * _tiling.l0_tk, a_gm.ptr.dtype, tla.AddressSpace.l0a, 512)
    l0b_ptr_0 = tla.allocate(_tiling.l0_tk * _tiling.l0_tn, b_gm.ptr.dtype, tla.AddressSpace.l0b, 512)
    l0a_ptr_1 = tla.allocate(_tiling.l0_tm * _tiling.l0_tk, a_gm.ptr.dtype, tla.AddressSpace.l0a, 512)
    l0b_ptr_1 = tla.allocate(_tiling.l0_tk * _tiling.l0_tn, b_gm.ptr.dtype, tla.AddressSpace.l0b, 512)
    remote_ub_ptr_0 = tla.allocate(_tiling.comm_tile_m * _tiling.comm_tile_n, tla.Float32, tla.AddressSpace.ub, 256)
    remote_ub_ptr_1 = tla.allocate(_tiling.comm_tile_m * _tiling.comm_tile_n, tla.Float32, tla.AddressSpace.ub, 256)
    l0c_ptr = tla.allocate(_tiling.l0_tm * _tiling.l0_tn, tla.Float32, tla.AddressSpace.l0c, 512)

    buffers = _MmadBuffers(
        l1a_ptr_0=l1a_ptr_0,
        l1a_ptr_1=l1a_ptr_1,
        l1b_ptr_0=l1b_ptr_0,
        l1b_ptr_1=l1b_ptr_1,
        l0a_ptr_0=l0a_ptr_0,
        l0a_ptr_1=l0a_ptr_1,
        l0b_ptr_0=l0b_ptr_0,
        l0b_ptr_1=l0b_ptr_1,
        l0c_ptr=l0c_ptr,
    )
    flags = _MmadFlags(
        l1a_loaded_0=l1a_loaded_0,
        l1a_loaded_1=l1a_loaded_1,
        l1b_loaded_0=l1b_loaded_0,
        l1b_loaded_1=l1b_loaded_1,
        l1a_available_0=l1a_available_0,
        l1a_available_1=l1a_available_1,
        l1b_available_0=l1b_available_0,
        l1b_available_1=l1b_available_1,
        l0a_available_0=l0a_available_0,
        l0a_available_1=l0a_available_1,
        l0b_available_0=l0b_available_0,
        l0b_available_1=l0b_available_1,
        l0_ab_loaded=l0_ab_loaded,
        l0c_ready=l0c_ready,
        l0c_available=l0c_available,
    )

    comm_buffers = _CommBuffers(
        remote_ub_ptr_0=remote_ub_ptr_0,
        remote_ub_ptr_1=remote_ub_ptr_1,
    )
    comm_flags = _CommFlags(
        remote_available_0=remote_available_0,
        remote_available_1=remote_available_1,
        remote_submit_0=remote_submit_0,
        remote_submit_1=remote_submit_1,
    )

    matmul_block_shape = tla.make_shape(_tiling.l1_tm, _tiling.l1_tn)
    a_shard_shape = tla.make_shape(shard_m, k)
    a_block_shape = tla.make_shape(_tiling.l1_tm, k)
    b_block_shape = tla.make_shape(k, _tiling.l1_tn)
    comm_block_shape = tla.make_shape(_tiling.comm_block_m, _tiling.comm_block_n)
    comm_blocks_m = _tiling.l1_tm // _tiling.comm_block_m
    comm_blocks_n = _tiling.l1_tn // _tiling.comm_block_n
    comm_blocks_per_mmad = comm_blocks_m * comm_blocks_n

    aic_id = tla.arch.block_idx()
    aic_count = tla.arch.block_num()
    blocks_per_comm_in_rank = aic_count * _tiling.comm_interval // world_size
    comm_loops = (blocks_per_shard + blocks_per_comm_in_rank - 1) // blocks_per_comm_in_rank

    with tla.cube():
        tla.set_flag(l1a_available_0)
        tla.set_flag(l1b_available_0)
        tla.set_flag(l1a_available_1)
        tla.set_flag(l1b_available_1)
        tla.set_flag(l0a_available_0)
        tla.set_flag(l0b_available_0)
        tla.set_flag(l0a_available_1)
        tla.set_flag(l0b_available_1)
        tla.set_flag(l0c_available)
        # Keep slot indices across K chunks, MN blocks and communication rounds.
        l1_buf_idx = 0
        l0_buf_idx = 0

        for comm_idx in tla.range(0, comm_loops, 1):
            stage_id = comm_idx % WORKSPACE_STAGES

            # Rounds 0 and 1 use different stages and can overlap their AIC
            # and AIV phases.  Starting from round 2, wait only for the AIV
            # consumer that previously used this same stage.
            if comm_idx >= WORKSPACE_STAGES:
                if stage_id == 0:
                    tla.cross_core_wait_flag(aiv_to_aic_stage_0, tla.arch.FIX)
                else:
                    tla.cross_core_wait_flag(aiv_to_aic_stage_1, tla.arch.FIX)

            blocks_this_rank = min(
                blocks_per_comm_in_rank,
                blocks_per_shard - comm_idx * blocks_per_comm_in_rank,
            )
            blocks_this_comm = blocks_this_rank * world_size

            # Flattened order is [dst shard, block_idx_in_rank].  The final
            # round uses only its valid blocks, while every dst
            # keeps blocks_per_comm_in_rank reserved workspace slots.  This is
            # the grid-stride MMAD scheduling used by the CatCCOS reference.
            for block_idx_in_comm in tla.range(aic_id, blocks_this_comm, aic_count):
                dst_rank = block_idx_in_comm // blocks_this_rank
                block_idx_in_rank = block_idx_in_comm - dst_rank * blocks_this_rank
                output_block_idx = comm_idx * blocks_per_comm_in_rank + block_idx_in_rank
                output_block_row, output_block_col = mmad_swizzle(
                    output_block_idx,
                    block_rows_per_shard,
                    block_cols_per_shard,
                    _swizzle,
                )
                workspace_block_idx = (stage_id * world_size + dst_rank) * blocks_per_comm_in_rank + block_idx_in_rank

                # Build the dst shard view first.  When shard_m is not aligned
                # to _tiling.l1_tm, deriving a global block row would cross the
                # boundary between adjacent shards.
                a_shard = tla.tile_view(
                    a_gm,
                    a_shard_shape,
                    tla.make_coord(dst_rank, 0),
                )
                a_block = tla.tile_view(
                    a_shard,
                    a_block_shape,
                    tla.make_coord(output_block_row, 0),
                )
                b_block = tla.tile_view(
                    b_gm,
                    b_block_shape,
                    tla.make_coord(0, output_block_col),
                )

                output_block = tla.tile_view(
                    output_shard,
                    matmul_block_shape,
                    tla.make_coord(
                        output_block_row,
                        output_block_col,
                    ),
                )

                # Keep the physical workspace slot at _tiling.l1_tm x _tiling.l1_tn, but
                # give the tensor the cropped logical origin.  L0C then uses
                # the same tail M/N as MMAD and Fixpipe writes only valid data.
                # The slot is used only when this block belongs to another
                # rank; the matching local slot remains untouched.
                workspace_layout = tla.make_layout(
                    matmul_block_shape,
                    tla.make_stride(_tiling.l1_tn, 1),
                    origin_shape=tla.make_shape(a_block.origin_shape[0], b_block.origin_shape[1]),
                    layoutTag=tla.arch.RowMajor,
                )
                workspace_block = tla.make_tensor(
                    symmetric_workspace.ptr,
                    workspace_layout,
                    coord=tla.make_coord(
                        workspace_block_idx * _tiling.l1_tm,
                        0,
                    ),
                )

                if dst_rank == rank:
                    l1_buf_idx, l0_buf_idx = block_mmad(
                        a_block,
                        b_block,
                        output_block,
                        buffers,
                        flags,
                        l1_buf_idx,
                        l0_buf_idx,
                        _tiling,
                    )
                else:
                    l1_buf_idx, l0_buf_idx = block_mmad(
                        a_block,
                        b_block,
                        workspace_block,
                        buffers,
                        flags,
                        l1_buf_idx,
                        l0_buf_idx,
                        _tiling,
                    )

            # Signal both AIVs paired with this AIC on the produced stage.
            if stage_id == 0:
                tla.cross_core_set_flag(aic_to_aiv_stage_0, tla.arch.FIX)
            else:
                tla.cross_core_set_flag(aic_to_aiv_stage_1, tla.arch.FIX)

        tla.wait_flag(l1a_available_0)
        tla.wait_flag(l1b_available_0)
        tla.wait_flag(l1a_available_1)
        tla.wait_flag(l1b_available_1)
        tla.wait_flag(l0a_available_0)
        tla.wait_flag(l0b_available_0)
        tla.wait_flag(l0a_available_1)
        tla.wait_flag(l0b_available_1)
        tla.wait_flag(l0c_available)

    with tla.vector():
        tla.set_flag(remote_available_0)
        tla.set_flag(remote_available_1)
        remote_buf_idx = 0
        aiv_id = tla.arch.sub_block_idx()
        # TLA block_idx() is the logical core ID shared by both AIV sub-blocks.
        logical_aiv_id = aic_id
        # Only sub-block 0 transfers data. All AIVs still participate in
        # workspace handshakes and global barriers.
        aiv_count = _tiling.comm_data_split * _tiling.comm_npu_split

        for comm_idx in tla.range(0, comm_loops, 1):
            stage_id = comm_idx % WORKSPACE_STAGES
            if stage_id == 0:
                tla.cross_core_wait_flag(aic_to_aiv_stage_0, tla.arch.MTE2)
            else:
                tla.cross_core_wait_flag(aic_to_aiv_stage_1, tla.arch.MTE2)

            # All ranks publish this round's remote workspace blocks and
            # finish their direct local-GM stores before remote reads/adds.
            tla.pipe_barrier(tla.pipes.ALL)
            tla_dist.barrier_all()

            blocks_this_rank = min(
                blocks_per_comm_in_rank,
                blocks_per_shard - comm_idx * blocks_per_comm_in_rank,
            )
            blocks_this_comm = blocks_this_rank * world_size

            # Swizzle [communication block, peer]. Each MMAD workspace slot
            # contains comm_blocks_per_mmad independently scheduled blocks.
            comm_blocks_this_rank = blocks_this_rank * comm_blocks_per_mmad
            comm_tasks = comm_blocks_this_rank * world_size
            comm_offset = _tiling.comm_data_split * _swizzle.COMM_SWIZZLE_OFFSET
            peer_stride = world_size // _tiling.comm_npu_split
            if tla.const_expr(_swizzle.COMM_SWIZZLE_DIRECTION == 1):
                comm_offset = _tiling.comm_npu_split * _swizzle.COMM_SWIZZLE_OFFSET
                peer_stride = world_size // comm_offset
            if tla.const_expr(_swizzle.IS_DETERMINISTIC):
                comm_offset = aiv_count * _swizzle.COMM_SWIZZLE_OFFSET
                peer_stride = world_size
                comm_tasks = ((comm_blocks_this_rank + comm_offset - 1) // comm_offset) * comm_offset * world_size

            comm_start = comm_tasks
            if aiv_id == 0 and logical_aiv_id < aiv_count:
                comm_start = logical_aiv_id
            for comm_task in tla.range(comm_start, comm_tasks, aiv_count):
                comm_block_idx, peer = comm_swizzle(
                    comm_task,
                    comm_blocks_this_rank,
                    world_size,
                    comm_offset,
                    peer_stride,
                    _swizzle,
                )
                if comm_block_idx < comm_blocks_this_rank and peer != rank:
                    block_idx_in_rank = comm_block_idx // comm_blocks_per_mmad
                    comm_block_in_mmad = comm_block_idx % comm_blocks_per_mmad
                    output_block_idx = comm_idx * blocks_per_comm_in_rank + block_idx_in_rank
                    output_block_row, output_block_col = mmad_swizzle(
                        output_block_idx,
                        block_rows_per_shard,
                        block_cols_per_shard,
                        _swizzle,
                    )
                    workspace_block_idx = (stage_id * world_size + rank) * blocks_per_comm_in_rank + block_idx_in_rank

                    output_block = tla.tile_view(
                        output_shard,
                        matmul_block_shape,
                        tla.make_coord(
                            output_block_row,
                            output_block_col,
                        ),
                    )

                    workspace_layout = tla.make_layout(
                        matmul_block_shape,
                        tla.make_stride(_tiling.l1_tn, 1),
                        origin_shape=tla.make_shape(
                            output_block.origin_shape[0],
                            output_block.origin_shape[1],
                        ),
                        layoutTag=tla.arch.RowMajor,
                    )
                    workspace_block = tla.make_tensor(
                        symmetric_workspace.ptr,
                        workspace_layout,
                        coord=tla.make_coord(
                            workspace_block_idx * _tiling.l1_tm,
                            0,
                        ),
                    )

                    # A padded communication block may lie wholly beyond an
                    # M/N tail. Do not construct a view with a negative extent.
                    comm_row = comm_block_in_mmad // comm_blocks_n
                    comm_col = comm_block_in_mmad % comm_blocks_n
                    comm_row_valid = comm_row * _tiling.comm_block_m < output_block.origin_shape[0]
                    comm_col_valid = comm_col * _tiling.comm_block_n < output_block.origin_shape[1]
                    if comm_row_valid and comm_col_valid:
                        comm_block_coord = tla.make_coord(
                            comm_row,
                            comm_col,
                        )
                        output_comm_block = tla.tile_view(output_block, comm_block_shape, comm_block_coord)
                        workspace_comm_block = tla.tile_view(workspace_block, comm_block_shape, comm_block_coord)

                        remote_buf_idx = block_comm(
                            output_comm_block,
                            workspace_comm_block,
                            peer,
                            comm_buffers,
                            comm_flags,
                            remote_buf_idx,
                            _tiling,
                        )

            # No rank may overwrite a slot until every peer has read it.
            tla.pipe_barrier(tla.pipes.ALL)
            tla_dist.barrier_all()
            if comm_idx + WORKSPACE_STAGES < comm_loops:
                if stage_id == 0:
                    tla.cross_core_set_flag(aiv_to_aic_stage_0, tla.arch.MTE3)
                else:
                    tla.cross_core_set_flag(aiv_to_aic_stage_1, tla.arch.MTE3)

        tla.wait_flag(remote_available_0)
        tla.wait_flag(remote_available_1)


def _validate_tiling(
    tiling: TilingParams,
    swizzle: SwizzleParams,
    world_size: int,
    aic_count: int | None = None,
) -> None:
    for name in tiling.__dataclass_fields__:
        value = getattr(tiling, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (tiling.l0_tm, tiling.l0_tn) != (tiling.l1_tm, tiling.l1_tn):
        raise ValueError("this kernel requires L0 and L1 M/N tiles to match")
    if any(getattr(tiling, name) % 16 for name in ("l1_tm", "l1_tn", "l1_tk", "l0_tm", "l0_tn", "l0_tk")):
        raise ValueError("MMAD tile dimensions must be multiples of 16")
    if tiling.l1_tk % tiling.l0_tk:
        raise ValueError("l1_tk must be divisible by l0_tk")
    if tiling.l1_tm % tiling.comm_block_m or tiling.l1_tn % tiling.comm_block_n:
        raise ValueError("MMAD blocks must be divisible by communication blocks")
    if tiling.comm_block_m % tiling.comm_tile_m or tiling.comm_block_n % tiling.comm_tile_n:
        raise ValueError("communication blocks must be divisible by communication tiles")
    if tiling.comm_tile_n % 8:
        raise ValueError("comm_tile_n must be a multiple of 8 for aligned fp32 UB rows")
    rank_split = tiling.comm_npu_split
    if swizzle.COMM_SWIZZLE_DIRECTION == 1:
        rank_split *= swizzle.COMM_SWIZZLE_OFFSET
    if world_size % rank_split:
        raise ValueError("the communication rank split must divide world_size")
    if aic_count is not None:
        if aic_count * tiling.comm_interval % world_size:
            raise ValueError("aic_count * comm_interval must be divisible by world_size")
        if tiling.comm_data_split * tiling.comm_npu_split > aic_count:
            raise ValueError("communication core split exceeds the available AIC count")


def _workspace_shape(tiling: TilingParams, aic_count: int) -> tuple[int, int]:
    return (
        WORKSPACE_STAGES * aic_count * tiling.comm_interval * tiling.l1_tm,
        tiling.l1_tn,
    )


def _validate_swizzle(swizzle: SwizzleParams, world_size: int) -> None:
    for name in ("MMAD_SWIZZLE_DIRECTION", "COMM_SWIZZLE_DIRECTION"):
        if getattr(swizzle, name) not in (0, 1):
            raise ValueError(f"{name} must be 0 or 1")
    for name in ("MMAD_SWIZZLE_OFFSET", "COMM_SWIZZLE_OFFSET"):
        value = getattr(swizzle, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if swizzle.IS_DETERMINISTIC and swizzle.COMM_SWIZZLE_DIRECTION != 0:
        raise ValueError("deterministic communication requires direction 0")
    if swizzle.COMM_SWIZZLE_DIRECTION == 1 and world_size % swizzle.COMM_SWIZZLE_OFFSET != 0:
        raise ValueError("COMM_SWIZZLE_OFFSET must divide world_size for direction 1")


def _launch_context(args: argparse.Namespace) -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", args.rank))
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    device = int(os.environ.get("LOCAL_RANK", args.device))
    for name in ("m", "n", "k"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("world_size must be positive and rank must be in [0, world_size)")
    if args.m % world_size:
        raise ValueError("m must be divisible by world_size")
    if args.block_num != -1 and args.block_num <= 0:
        raise ValueError("block_num must be -1 (all AICs) or positive")
    if args.dtype_a != args.dtype_b:
        raise ValueError("A and B must have the same dtype")
    return rank, world_size, device


def _to_tla(tensor: Any, layout: str = "row") -> Any:
    # Match the reference example: physical column-major storage is the
    # contiguous transpose, interpreted with a ColumnMajor layout tag.
    if layout == "col":
        tensor = tensor.permute(1, 0).contiguous()
    else:
        tensor = tensor.contiguous()
    tag = tla.arch.RowMajor if layout == "row" else tla.arch.ColumnMajor
    return tla.from_dlpack(tensor, layout_tag=tag).mark_layout_dynamic()


def _compile_only_args(
    swizzle: SwizzleParams = SWIZZLE,
    tiling: TilingParams = TILING,
    *,
    args: argparse.Namespace | None = None,
) -> tuple[Any, ...]:
    if args is None:
        args = _argument_parser().parse_args([])
    rank, world_size, _ = _launch_context(args)
    _validate_swizzle(swizzle, world_size)
    _validate_tiling(tiling, swizzle, world_size)
    # Without a device query, construct valid metadata for the requested core
    # split. An explicit --block-num is used exactly as in runtime mode.
    aic_count = args.block_num
    if aic_count == -1:
        aic_count = tiling.comm_data_split * tiling.comm_npu_split * world_size
    _validate_tiling(tiling, swizzle, world_size, aic_count)
    workspace_m, workspace_n = _workspace_shape(tiling, aic_count)
    dtypes = {"f16": tla.Float16, "bf16": tla.BFloat16, "f32": tla.Float32}

    def tensor(rows: int, columns: int, dtype: Any, layout: str = "row") -> Any:
        stride = (columns, 1) if layout == "row" else (1, rows)
        tag = tla.arch.RowMajor if layout == "row" else tla.arch.ColumnMajor
        return tla.make_fake_tensor(
            dtype,
            (rows, columns),
            stride,
            origin_shape=(rows, columns),
            coord=(0, 0),
            layout_tag=tag,
        ).mark_layout_dynamic()

    return (
        tensor(args.m, args.k, dtypes[args.dtype_a], args.layout_a),
        tensor(args.k, args.n, dtypes[args.dtype_b], args.layout_b),
        tensor(workspace_m, workspace_n, tla.Float32),
        tensor(args.m // world_size, args.n, dtypes[args.dtype_c]),
        tla.Int32(rank),
        tla.Int32(world_size),
        tiling,
        swizzle,
    )


def dump_tlair(
    swizzle: SwizzleParams = SWIZZLE,
    tiling: TilingParams = TILING,
    *,
    args: argparse.Namespace | None = None,
) -> str:
    """Return compile-only TLA IR without importing torch or ACLSHMEM."""
    return matmul_reduce_scatter.dump_mlir(type_args=_compile_only_args(swizzle, tiling, args=args))


def _broadcast_shmem_uid(*, rank: int, torch: Any, dist: Any) -> bytes:
    uid_tensor = torch.zeros(SHMEM_UID_BUFFER_SIZE, dtype=torch.uint8)
    if rank == 0:
        unique_id = tla_dist.distributed.get_unique_id()
        if not isinstance(unique_id, (bytes, bytearray, memoryview)):
            raise tla_dist.distributed.ShmemRuntimeError(
                f"ACLSHMEM unique id must be bytes-like, got {type(unique_id).__name__}"
            )
        uid_bytes = bytes(unique_id)
        if not uid_bytes or len(uid_bytes) > SHMEM_UID_BUFFER_SIZE:
            raise tla_dist.distributed.ShmemRuntimeError(
                f"invalid ACLSHMEM unique-id size: capacity={SHMEM_UID_BUFFER_SIZE}, actual={len(uid_bytes)}"
            )
        uid_tensor[: len(uid_bytes)].copy_(torch.tensor(list(uid_bytes), dtype=torch.uint8))
    dist.broadcast(uid_tensor, src=0)
    return bytes(uid_tensor.tolist())


def run(
    args: argparse.Namespace,
    *,
    tiling: TilingParams = TILING,
    warmup_iters: int = 0,
    repeat_iters: int = 1,
    print_tensors: bool = True,
    result_dir: Path | None = None,
    synchronize_launches: bool = False,
) -> int:
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    rank, world_size, local_rank = _launch_context(args)
    shard_m = args.m // world_size
    if synchronize_launches and warmup_iters < 1:
        raise ValueError("synchronized launches require at least one warmup iteration")

    _validate_swizzle(SWIZZLE, world_size)
    _validate_tiling(tiling, SWIZZLE, world_size)
    torch.npu.set_device(local_rank)
    dist.init_process_group(
        backend=CONTROL_PLANE_BACKEND,
        rank=rank,
        world_size=world_size,
    )
    shmem_runtime = None
    try:
        tla_dist.distributed.configure_store_tls(False, "")
        unique_id = _broadcast_shmem_uid(rank=rank, torch=torch, dist=dist)
        shmem_runtime = tla_dist.distributed.ShmemRuntime(
            uid=unique_id,
            rank=rank,
            world_size=world_size,
            heap_size=args.heap_size,
            device_id=local_rank,
        ).initialize()

        # torch_npu owns ACL and the current launch stream in this example.
        # Avoid a second tla.initialize()/finalize() lifecycle, which would
        # finalize ACL before torch_npu exits.
        device_properties = torch.npu.get_device_properties(int(local_rank))
        available_aics = int(device_properties.cube_core_num)
        aic_count = available_aics if args.block_num == -1 else args.block_num
        if aic_count > available_aics:
            raise ValueError("block_num exceeds the device's AIC count")
        peer_aic_counts: list[int | None] = [None] * world_size
        dist.all_gather_object(peer_aic_counts, aic_count)
        if any(peer_aic_count != aic_count for peer_aic_count in peer_aic_counts):
            raise RuntimeError(
                f"all ranks must expose the same AIC count for symmetric workspace scheduling, got {peer_aic_counts}"
            )
        _validate_tiling(tiling, SWIZZLE, world_size, aic_count)
        blocks_per_comm = aic_count * tiling.comm_interval
        symmetric_workspace_m, workspace_n = _workspace_shape(tiling, aic_count)
        print(
            f"rank={rank} local_rank={local_rank} "
            f"mnk=({args.m},{args.n},{args.k}) "
            f"layout={args.layout_a}/{args.layout_b} "
            f"dtype={args.dtype_a}/{args.dtype_b}/{args.dtype_c} "
            f"aic_count={aic_count} "
            f"blocks_per_comm={blocks_per_comm} "
            "symmetric_workspace_shape="
            f"({symmetric_workspace_m}, {workspace_n})",
            flush=True,
        )

        torch.npu.manual_seed(args.seed + rank)
        dtypes = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}
        a = torch.empty((args.m, args.k), dtype=dtypes[args.dtype_a], device="npu")
        b = torch.empty((args.k, args.n), dtype=dtypes[args.dtype_b], device="npu")
        a.uniform_(-1.0, 1.0)
        b.uniform_(-1.0, 1.0)
        symmetric_workspace = shmem_runtime.symmetric_tensor(
            (symmetric_workspace_m, workspace_n),
            torch.float32,
        )
        output = torch.zeros((shard_m, args.n), dtype=dtypes[args.dtype_c], device="npu")

        ta = _to_tla(a, args.layout_a)
        tb = _to_tla(b, args.layout_b)
        tworkspace = _to_tla(symmetric_workspace)
        tout = _to_tla(output)
        rank_arg = tla.Int32(rank)
        world_arg = tla.Int32(world_size)
        cache_dir = str(Path(args.cache_dir).expanduser().resolve())

        compile_args = (
            matmul_reduce_scatter,
            ta,
            tb,
            tworkspace,
            tout,
            rank_arg,
            world_arg,
            tiling,
            SWIZZLE,
        )
        compile_kwargs = {
            "arch_scope": "aic.c310",
            "cache_dir": cache_dir,
        }
        if rank == 0:
            artifact = tla_dist.compile(
                *compile_args,
                **compile_kwargs,
                force_recompile=args.force_recompile,
            )

        # Serialize access to the shared on-disk compilation cache.
        dist.barrier()
        if rank != 0:
            artifact = tla_dist.compile(
                *compile_args,
                **compile_kwargs,
                force_recompile=False,
            )

        # Each rank issues exactly warmup_iters + repeat_iters target kernels.
        torch.npu.synchronize()
        if synchronize_launches:
            # Load the compiled binary in the first warmup, before graph capture.
            # The remaining calls replay a graph containing both the device
            # barrier and kernel, so Python launch delay cannot fall between them.
            shmem_runtime.barrier_all()
            artifact(
                ta,
                tb,
                tworkspace,
                tout,
                rank_arg,
                world_arg,
                tiling,
                SWIZZLE,
                block_num=aic_count,
            )
            torch.npu.synchronize()
            dist.barrier()
            launch_stream = torch.npu.Stream()
            launch_graph = torch.npu.NPUGraph()
            with torch.npu.graph(launch_graph, stream=launch_stream, capture_error_mode="relaxed"):
                shmem_runtime.barrier_all_on_stream(launch_stream.npu_stream)
                artifact(
                    ta,
                    tb,
                    tworkspace,
                    tout,
                    rank_arg,
                    world_arg,
                    tiling,
                    SWIZZLE,
                    block_num=aic_count,
                    stream=launch_stream.npu_stream,
                )
            dist.barrier()
            for _ in range(warmup_iters + repeat_iters - 1):
                with torch.npu.stream(launch_stream):
                    launch_graph.replay()
                launch_stream.synchronize()
        else:
            for _ in range(warmup_iters + repeat_iters):
                shmem_runtime.barrier_all()
                artifact(
                    ta,
                    tb,
                    tworkspace,
                    tout,
                    rank_arg,
                    world_arg,
                    tiling,
                    SWIZZLE,
                    block_num=aic_count,
                )
                torch.npu.synchronize()
        dist.barrier()

        local_a = a.float().cpu()
        local_b = b.float().cpu()
        all_a = [torch.empty_like(local_a) for _ in range(world_size)]
        all_b = [torch.empty_like(local_b) for _ in range(world_size)]
        dist.all_gather(all_a, local_a)
        dist.all_gather(all_b, local_b)
        expected_full = sum(peer_a @ peer_b for peer_a, peer_b in zip(all_a, all_b))
        expected = expected_full[
            rank * shard_m : (rank + 1) * shard_m,
            :,
        ]
        actual = output.cpu()
        difference = actual - expected
        max_abs_error = float(difference.abs().max())
        passed = bool(torch.allclose(actual, expected, rtol=1 / 64, atol=1 / 16))

        # Serialize matrix output across ranks so concurrent torchrun workers do
        # not interleave rows and make the numerical result difficult to read.
        for print_rank in range(world_size):
            dist.barrier()
            if rank == print_rank:
                if print_tensors:
                    print(f"rank={rank} actual output:\n{actual}", flush=True)
                    print(f"rank={rank} expected output:\n{expected}", flush=True)
                    print(f"rank={rank} actual - expected:\n{difference}", flush=True)
                print(
                    f"rank={rank} passed={passed} "
                    f"max_abs_error={max_abs_error:.6g} "
                    f"kernel.o={artifact.kernel_binary_path}",
                    flush=True,
                )
        if result_dir is not None:
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / f"rank_{rank}.json").write_text(
                json.dumps(
                    {
                        "rank": rank,
                        "world_size": world_size,
                        "passed": passed,
                        "max_abs_error": max_abs_error,
                        "aic_count": aic_count,
                        "warmup_iters": warmup_iters,
                        "repeat_iters": repeat_iters,
                        "kernel_name": "matmul_reduce_scatter",
                        "kernel_binary": str(artifact.kernel_binary_path),
                    },
                    indent=2,
                )
            )
        dist.barrier()
        return 0 if passed else 1
    finally:
        try:
            # Release SHMEM while torch_npu still owns a live device context.
            if shmem_runtime is not None:
                shmem_runtime.finalize()
        finally:
            dist.destroy_process_group()


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=2060)
    parser.add_argument("--n", type=int, default=2516)
    parser.add_argument("--k", type=int, default=166)
    parser.add_argument("--layout-a", choices=("row", "col"), default="row")
    parser.add_argument("--layout-b", choices=("row", "col"), default="row")
    parser.add_argument("--dtype-a", choices=("f16", "bf16"), default="f16")
    parser.add_argument("--dtype-b", choices=("f16", "bf16"), default="f16")
    parser.add_argument("--dtype-c", choices=("f32",), default="f32")
    parser.add_argument(
        "--block-num",
        type=int,
        default=-1,
        help="number of AICs to launch; -1 uses all available AICs",
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--heap-size", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--dump-tlair", action="store_true")
    return parser


def main() -> int:
    parser = _argument_parser()
    args = parser.parse_args()
    try:
        _launch_context(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dump_tlair:
        print(dump_tlair(args=args))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
