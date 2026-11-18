
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPERATOR_HOST_H
#define OPERATOR_HOST_H

#include "allgather_matmul/allgather_matmul_host.h"
#include "allgather_matmul_dequant/allgather_matmul_dequant_host.h"
#include "allgather_matmul_dequant_bias/allgather_matmul_dequant_bias_host.h"
#include "allgather_matmul_rdma/allgather_matmul_rdma_host.h"
#include "allgather_matmul_remote_read/allgather_matmul_remote_read_host.h"
#include "allgather_matmul_with_gather_result/allgather_matmul_with_gather_result_host.h"
#include "alltoallv_gmm_v2/alltoallv_gmm_v2_host.h"
#include "alltoallv_grouped_matmul/alltoallv_grouped_matmul_host.h"
#include "grouped_matmul_alltoallv/grouped_matmul_alltoallv_host.h"
#include "matmul_allreduce/matmul_allreduce_host.h"
#include "matmul_dequant_reduce_scatter_write/matmul_dequant_reduce_scatter_write_host.h"
#include "matmul_reduce_scatter/matmul_reduce_scatter_host.h"

#ifdef CATCCOS_ENABLE_A5_BUILD
#include "ascend950_allgather_matmul/ascend950_allgather_matmul_host.h"
#include "ascend950_allgather_matmul_udma/ascend950_allgather_matmul_udma_host.h"
#include "ascend950_alltoallv_grouped_matmul/ascend950_alltoallv_grouped_matmul_host.h"
#include "ascend950_fp4_mx_allgather_matmul/ascend950_fp4_mx_allgather_matmul_host.h"
#include "ascend950_fp4_mx_alltoallv_grouped_matmul/ascend950_fp4_mx_alltoallv_grouped_matmul_host.h"
#include "ascend950_fp4_mx_grouped_matmul_alltoallv/ascend950_fp4_mx_grouped_matmul_alltoallv_host.h"
#include "ascend950_fp4_mx_matmul_reduce_scatter/ascend950_fp4_mx_matmul_reduce_scatter_host.h"
#include "ascend950_fp8_mx_allgather_matmul/ascend950_fp8_mx_allgather_matmul_host.h"
#include "ascend950_fp8_mx_alltoallv_grouped_matmul/ascend950_fp8_mx_alltoallv_grouped_matmul_host.h"
#include "ascend950_fp8_mx_grouped_matmul_alltoallv/ascend950_fp8_mx_grouped_matmul_alltoallv_host.h"
#include "ascend950_grouped_matmul_alltoallv/ascend950_grouped_matmul_alltoallv_host.h"
#include "ascend950_matmul_reduce_scatter/ascend950_matmul_reduce_scatter_host.h"
#include "ascend950_mx_quant_allgather/mx_quant_allgather_host.h"
#include "ascend950_mxfp8_alltoall_matmul_split_k_urma/ascend950_mxfp8_alltoall_matmul_split_k_urma_host.h"
#include "ascend950_mxfp8_matmul_alltoall/ascend950_mxfp8_matmul_alltoall_host.h"
#include "ascend950_matmul_reduce_scatter_udma/ascend950_matmul_reduce_scatter_udma_host.h"
#endif

#endif // OPERATOR_HOST_H
