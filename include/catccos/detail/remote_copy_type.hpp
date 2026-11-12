
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_DETAIL_REMOTE_COPY_TYPE_HPP
#define CATCCOS_DETAIL_REMOTE_COPY_TYPE_HPP

namespace Catccos::detail {

enum class CopyDirect { Put, Get };
enum class CopyTransport { Mte, Rdma, Udma };

} // namespace Catccos::detail

#endif // CATCCOS_DETAIL_REMOTE_COPY_TYPE_HPP
