# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared parameters with defaults from examples/ascend950_matmul_reduce_scatter."""

from __future__ import annotations

from dataclasses import dataclass

from catccos import tla_dist as tla


@dataclass(frozen=True)
class TilingParams:
    # Ascend950MatmulReduceScatterConfig_M0_128.
    l1_tm: tla.Constexpr[int] = 128
    l1_tn: tla.Constexpr[int] = 256
    l1_tk: tla.Constexpr[int] = 256
    l0_tm: tla.Constexpr[int] = 128
    l0_tn: tla.Constexpr[int] = 256
    l0_tk: tla.Constexpr[int] = 64

    # Communication shapes passed to DeviceOp::Arguments. The host's
    # commTileM=64 is halved for the two AIV sub-blocks, giving tile M=32.
    comm_block_m: tla.Constexpr[int] = 64
    comm_block_n: tla.Constexpr[int] = 256
    comm_tile_m: tla.Constexpr[int] = 32
    comm_tile_n: tla.Constexpr[int] = 256

    # A round contains aic_count * comm_interval MMAD blocks across all ranks.
    # This product must be divisible by rank_size.
    comm_interval: tla.Constexpr[int] = 10
    # commCoreSplit = (commDataSplit, commNpuSplit).
    comm_data_split: tla.Constexpr[int] = 20
    comm_npu_split: tla.Constexpr[int] = 1


@dataclass(frozen=True)
class SwizzleParams:
    # Shared by GemmIdentityBlockSwizzle and the communication destination
    # remapper; both must use the same MMAD traversal.
    MMAD_SWIZZLE_DIRECTION: tla.Constexpr[int] = 1
    MMAD_SWIZZLE_OFFSET: tla.Constexpr[int] = 7
    # BlockCommSchedulerReduceScatter requires direction 0 when deterministic.
    COMM_SWIZZLE_DIRECTION: tla.Constexpr[int] = 0
    COMM_SWIZZLE_OFFSET: tla.Constexpr[int] = 1
    IS_DETERMINISTIC: tla.Constexpr[bool] = False
