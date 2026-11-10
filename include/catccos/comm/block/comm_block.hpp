
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_COMM_BLOCK_HPP
#define CATCCOS_COMM_BLOCK_HPP

#include "catccos/catccos.hpp"

namespace Catccos::Comm::Block {

template <class DispatchPolicy, class... Args>
class CommBlock {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "Could not find an epilogue specialization");
};

} // namespace Catccos::Comm::Block

#include "catccos/comm/block/comm_block_to_local_mem.hpp"
#include "catccos/comm/block/comm_block_to_share_mem.hpp"
#include "catccos/comm/block/comm_block_remote_copy.hpp"
#include "catccos/comm/block/comm_block_hierarchical_copy.hpp"
#include "catccos/comm/block/comm_block_remote_chunk_copy.hpp"
#include "catccos/comm/block/comm_block_local_copy.hpp"
#include "catccos/comm/block/comm_block_rdma_copy.hpp"
#endif // CATCCOS_COMM_BLOCK_HPP
