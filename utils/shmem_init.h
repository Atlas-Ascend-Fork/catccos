
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_INIT_H
#define SHMEM_INIT_H

#include <algorithm>
#include <cstring>
#include "shmem.h"

inline int32_t set_attr(
    int32_t my_pe, int32_t n_pes, uint64_t local_mem_size, const char* ip_port, aclshmemx_init_attr_t* attributes,
    aclshmemx_uniqueid_t* flag_uid)
{
    size_t ip_len = 0;
    if (ip_port != nullptr) {
        ip_len = std::min(strlen(ip_port), sizeof(attributes->ip_port) - 1);
        std::copy_n(ip_port, ip_len, attributes->ip_port);
        if (attributes->ip_port[0] == '\0') {
            return ACLSHMEM_INVALID_VALUE;
        }
    }

    attributes->my_pe = my_pe;
    attributes->n_pes = n_pes;
    attributes->ip_port[ip_len] = '\0';
    attributes->local_mem_size = local_mem_size;
    // The SDK owns the attribute ABI version. Replacing option_attr with a
    // format-1 aggregate makes format-2 SDKs ignore runtime_config.
    attributes->option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_MTE;
    attributes->option_attr.shm_init_timeout = DEFAULT_TIMEOUT;
    attributes->option_attr.shm_create_timeout = DEFAULT_TIMEOUT;
    attributes->option_attr.control_operation_timeout = DEFAULT_TIMEOUT;
    flag_uid->my_pe = my_pe;
    flag_uid->n_pes = n_pes;
    attributes->comm_args = reinterpret_cast<void*>(flag_uid);
    return ACLSHMEM_SUCCESS;
}

#endif // SHMEM_INIT_H
