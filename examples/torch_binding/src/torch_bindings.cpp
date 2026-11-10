/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <optional>
#include <string>
#include <torch/torch.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include "shmem.h"
#include "catccos_torch_kernel.h"
#include "utils.h"
#include "shmem_init.h"

#ifdef CATCCOS_TORCH_A5
#include "catccos/detail/ascend950_moe_ready.hpp"
#include "catccos/detail/ascend950_moe_ub.h"
#else
#include "info.h"
#endif

namespace {

struct CatccosTorchOpState {
    bool initialized = false;
    void* symmPtr = nullptr;
    size_t symmBytes = 0;
    int64_t rankId = -1;
    int64_t rankSize = 0;
#ifdef CATCCOS_TORCH_A5
    at::Tensor workspace;
    std::optional<c10::Device> device;
    std::optional<c10_npu::NPUStream> lastLaunchStream;
#endif
};

CatccosTorchOpState& get_catccos_state()
{
    static CatccosTorchOpState state;
    return state;
}

#ifdef CATCCOS_TORCH_A5
void synchronize_catccos_stream(const c10_npu::NPUStream& stream)
{
    // stream() drains torch_npu's task queue before the ACL wait.
    aclrtStream acl_stream = stream.stream();
    TORCH_CHECK(acl_stream != nullptr, "failed to drain CatCCOS NPU stream");
    auto status = aclrtSynchronizeStream(acl_stream);
    TORCH_CHECK(status == ACL_SUCCESS, "CatCCOS stream synchronization failed, ret=", status);
}
#endif

void* get_catccos_symm_ptr(size_t bytes)
{
    auto& state = get_catccos_state();
    TORCH_CHECK(
        state.initialized, "catccos TORCH_LIBRARY not initialized, "
                           "call torch.ops.catccos.init(...) first");

    if (state.symmPtr == nullptr) {
        state.symmPtr = shmem_malloc(bytes);
        TORCH_CHECK(state.symmPtr != nullptr, "catccos TORCH_LIBRARY shmem_malloc failed, bytes=", bytes);
        state.symmBytes = bytes;

        auto ret = aclrtMemset(state.symmPtr, bytes, 0, bytes);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtMemset failed, ret=", ret);
    } else {
        TORCH_CHECK(
            state.symmBytes >= bytes, "existing symm buffer too small, existing=", state.symmBytes,
            ", required=", bytes);
    }

    return state.symmPtr;
}

uint64_t get_catccos_ffts()
{
    TORCH_CHECK(
        get_catccos_state().initialized, "catccos TORCH_LIBRARY not initialized, "
                                         "call torch.ops.catccos.init(...) first");

    uint64_t addr = shmemx_get_ffts_config();
#ifndef CATCCOS_TORCH_A5
    TORCH_CHECK(addr != 0, "shmemx_get_ffts_config returned 0");
#endif
    return addr;
}

} // namespace

namespace catccos {

int64_t init(int64_t rank_id, int64_t rank_size, int64_t local_mem_size, const std::string& ip_port)
{
    auto& state = get_catccos_state();
    if (state.initialized) {
#ifdef CATCCOS_TORCH_A5
        TORCH_CHECK(
            rank_id == state.rankId && rank_size == state.rankSize,
            "CatCCOS is already initialized with a different rank configuration");
#endif
        return 0;
    }

    TORCH_CHECK(rank_id >= 0, "invalid rank_id: ", rank_id);
    TORCH_CHECK(rank_size > 0, "invalid rank_size: ", rank_size);
#ifdef CATCCOS_TORCH_A5
    TORCH_CHECK(rank_id < rank_size && rank_size <= INT32_MAX, "invalid rank configuration");
    const auto device = c10_npu::getCurrentNPUStream().device();
#endif
    TORCH_CHECK(local_mem_size > 0, "invalid local_mem_size: ", local_mem_size);
    TORCH_CHECK(!ip_port.empty(), "ip_port is empty");

    int64_t status = aclshmemx_set_conf_store_tls(false, nullptr, 0);
    TORCH_CHECK(status == ACLSHMEM_SUCCESS, "aclshmemx_set_conf_store_tls failed: ", status);

    aclshmemx_uniqueid_t default_flag_uid{};
    aclshmemx_init_attr_t attributes{};
    int32_t ret = set_attr(
        static_cast<int32_t>(rank_id), static_cast<int32_t>(rank_size), static_cast<uint64_t>(local_mem_size),
        ip_port.c_str(), &attributes, &default_flag_uid);
    TORCH_CHECK(ret == ACLSHMEM_SUCCESS, "set_attr failed, ret=", ret);

    status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    TORCH_CHECK(status == ACLSHMEM_SUCCESS, "aclshmemx_init_attr failed: ", status);

    state.initialized = true;
    state.rankId = rank_id;
    state.rankSize = rank_size;
#ifdef CATCCOS_TORCH_A5
    state.device = device;
#endif
    return 0;
}

#ifndef CATCCOS_TORCH_A5
at::Tensor allgather_matmul(const at::Tensor& a, const at::Tensor& b, int64_t rank_size)
{
    c10_npu::OptionalNPUGuard guard(a.device());

    auto& state = get_catccos_state();
    TORCH_CHECK(
        state.initialized, "catccos TORCH_LIBRARY not initialized, "
                           "call torch.ops.catccos.init(...) first");
    TORCH_CHECK(rank_size == state.rankSize, "rank_size mismatch: input=", rank_size, ", initialized=", state.rankSize);

    TORCH_CHECK(a.dtype() == at::kHalf, "Only float16 supported for a, got ", a.dtype());
    TORCH_CHECK(b.dtype() == at::kHalf, "Only float16 supported for b, got ", b.dtype());
    TORCH_CHECK(a.device().type() == c10::DeviceType::PrivateUse1, "Only NPU supported for a, got ", a.device());
    TORCH_CHECK(b.device().type() == c10::DeviceType::PrivateUse1, "Only NPU supported for b, got ", b.device());
    TORCH_CHECK(a.device() == b.device(), "a and b must be on same device");
    TORCH_CHECK(a.dim() == 2, "a must be 2D, got dim=", a.dim());
    TORCH_CHECK(b.dim() == 2, "b must be 2D, got dim=", b.dim());

    int64_t m = a.size(0);
    int64_t k = a.size(1);
    int64_t n = b.size(1);
    TORCH_CHECK(b.size(0) == k, "A/K shape mismatch, a.size(1)=", k, ", b.size(0)=", b.size(0));
    TORCH_CHECK(
        m <= UINT32_MAX && n <= UINT32_MAX && k <= UINT32_MAX, "shape too large for uint32: m=", m, ", k=", k,
        ", n=", n);

    int32_t my_pe = shmem_my_pe();
    int32_t n_pes = shmem_n_pes();
    TORCH_CHECK(my_pe >= 0, "invalid my_pe: ", my_pe);
    TORCH_CHECK(n_pes > 0, "invalid n_pes: ", n_pes);
    TORCH_CHECK(rank_size == n_pes, "rank_size mismatch: input=", rank_size, ", shmem_n_pes=", n_pes);

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor c = at::empty({m * n_pes, n}, a.options()).contiguous();

    uint8_t* aPtr = static_cast<uint8_t*>(a_contig.data_ptr());
    uint8_t* bPtr = static_cast<uint8_t*>(b_contig.data_ptr());
    uint8_t* cPtr = static_cast<uint8_t*>(c.data_ptr());
    uint8_t* symmPtr = static_cast<uint8_t*>(get_catccos_symm_ptr(SHMEM_BUFF_BYTES));
    uint64_t ffts = get_catccos_ffts();
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    TORCH_CHECK(stream != nullptr, "NPU stream is null");

    at_npu::native::OpCommand cmd;
    cmd.Name("catccos_allgather_matmul");
    cmd.Input(a_contig);
    cmd.Input(b_contig);
    cmd.Output(c);
    cmd.SetCustomHandler([stream, ffts, aPtr, bPtr, cPtr, symmPtr, m, n, k, my_pe, n_pes]() -> int {
        CatccosKernel::catccos_allgather_matmul_wrapper(
            BLOCK_NUM, stream, ffts, aPtr, bPtr, cPtr, symmPtr, static_cast<uint32_t>(m), static_cast<uint32_t>(n),
            static_cast<uint32_t>(k), my_pe, n_pes);
        return 0;
    });
    cmd.Run();

    return c;
}
#else
at::Tensor ascend950_dispatch_ffn_combine(
    const at::Tensor& x, const at::Tensor& expert_idx, const at::Tensor& gate_weight, const at::Tensor& w1,
    const at::Tensor& w1_scale, const at::Tensor& w2, const at::Tensor& w2_scale)
{
    auto& state = get_catccos_state();
    TORCH_CHECK(state.initialized, "call torch.ops.catccos.init(...) first");
    TORCH_CHECK(x.device() == *state.device, "CatCCOS must run on its initialized NPU device");
    c10_npu::NPUGuard guard(x.device());

    struct Input {
        const at::Tensor& tensor;
        const char* name;
        at::ScalarType dtype;
        int64_t dims;
    };
    const Input inputs[] = {
        {x, "x", at::kBFloat16, 2},
        {expert_idx, "expert_idx", at::kInt, 2},
        {gate_weight, "gate_weight", at::kFloat, 2},
        {w1, "w1", at::ScalarType::Float8_e4m3fn, 3},
        {w1_scale, "w1_scale", at::ScalarType::Float8_e8m0fnu, 3},
        {w2, "w2", at::ScalarType::Float8_e4m3fn, 3},
        {w2_scale, "w2_scale", at::ScalarType::Float8_e8m0fnu, 3},
    };
    for (const auto& input : inputs) {
        const auto& tensor = input.tensor;
        TORCH_CHECK(tensor.device() == x.device(), input.name, " must be on the same device as x");
        TORCH_CHECK(tensor.scalar_type() == input.dtype, input.name, " must have dtype ", input.dtype);
        TORCH_CHECK(tensor.dim() == input.dims, input.name, " must be ", input.dims, "D");
        TORCH_CHECK(tensor.is_contiguous(), input.name, " must have contiguous strides");
        TORCH_CHECK(
            at_npu::native::get_npu_format(tensor) == ACL_FORMAT_ND, input.name,
            " must use ACL_FORMAT_ND storage; convert with torch_npu.npu_format_cast(tensor, 2)");
    }

    const int64_t m = x.size(0), k = x.size(1), n = w1.size(1);
    const int64_t top_k = expert_idx.size(1), local_experts = w1.size(0), rank_size = state.rankSize;
    TORCH_CHECK(m > 0 && k > 0 && n > 0, "M, K and N must be positive");
    TORCH_CHECK(k <= UINT32_MAX && n <= UINT32_MAX, "K or N exceeds the uint32 kernel limit");
    TORCH_CHECK(k % 256 == 0 && n % 512 == 0, "K must be divisible by 256 and N by 512");
    TORCH_CHECK(
        local_experts > 0 && local_experts <= INT32_MAX / rank_size,
        "total expert count exceeds the int32 kernel limit");
    TORCH_CHECK(top_k > 0 && top_k <= local_experts * rank_size, "invalid TopK");
    TORCH_CHECK(m <= INT32_MAX / rank_size / top_k, "routed row count exceeds the int32 kernel limit");

    const int64_t k2 = n / 2, scale_k = k / 32, scale_k2 = k2 / 32;
    TORCH_CHECK(expert_idx.size(0) == m && gate_weight.sizes() == expert_idx.sizes(), "expected [M, TopK] routing");
    TORCH_CHECK(w1.size(2) == k, "w1 must have shape [local_experts, N, K]");
    TORCH_CHECK(w1_scale.sizes() == at::IntArrayRef({local_experts, n, scale_k}), "invalid w1_scale shape");
    TORCH_CHECK(w2.sizes() == at::IntArrayRef({local_experts, k, k2}), "invalid w2 shape");
    TORCH_CHECK(w2_scale.sizes() == at::IntArrayRef({local_experts, k, scale_k2}), "invalid w2_scale shape");

    // SHMEM supports 48 AIVs, i.e. 24 mixed 1:2 blocks.
    constexpr uint32_t block_dim = 24;
    using Ub = Catccos::Detail::Ascend950MoeUb;
    TORCH_CHECK(
        Ub::Fits(m, k, top_k, 2 * block_dim), "shape-dependent buffers exceed Ascend950 UB capacity of ",
        Ub::kCapacityBytes, " bytes");
    TORCH_CHECK(shmem_my_pe() == state.rankId && shmem_n_pes() == rank_size, "SHMEM rank configuration mismatch");

    using Ready = Catccos::Detail::Ascend950MoeReady;
    const size_t expanded_m = m * top_k;
    const size_t rows = expanded_m * rank_size;
    const size_t meta_bytes = static_cast<size_t>(rank_size) * rank_size * local_experts * sizeof(int32_t);
    const size_t symmetric_capacity = CatccosKernel::catccos_ascend950_symmetric_bytes();
    TORCH_CHECK(
        meta_bytes <= symmetric_capacity && Ready::Align(expanded_m * k * sizeof(at::BFloat16)) +
                                                    Ready::Align(expanded_m * Ready::Align(k + scale_k, 32)) +
                                                    Ready::Align(meta_bytes) <=
                                                symmetric_capacity,
        "required symmetric buffer exceeds SHMEM_BUFF_BYTES");
    const size_t workspace_bytes =
        Ready::Align(Ready::Align(m, 256) * top_k * sizeof(int32_t)) + Ready::Align(rows * k * sizeof(at::BFloat16)) +
        2 * Ready::Align(meta_bytes) + Ready::Align(rows * k) + Ready::Align(rows * scale_k) +
        Ready::SwigluRegionBytes(rows * k2 * sizeof(at::BFloat16), local_experts) + Ready::Align(rows * k2) +
        Ready::Align(rows * scale_k2) +
        Ready::DispatchRegionBytes(rows * k * sizeof(at::BFloat16), rank_size, local_experts);
    TORCH_CHECK(workspace_bytes <= INT64_MAX, "workspace size exceeds the tensor size limit");

    const auto npu_stream = c10_npu::getCurrentNPUStream();
    aclrtStream stream = npu_stream.stream(false);
    TORCH_CHECK(stream != nullptr, "NPU stream is null");
    if (state.lastLaunchStream && *state.lastLaunchStream != npu_stream) {
        synchronize_catccos_stream(*state.lastLaunchStream);
    }

    const auto output = at_npu::native::empty_with_format({m, k}, x.options(), ACL_FORMAT_ND);
    if (!state.workspace.defined() || static_cast<size_t>(state.workspace.numel()) < workspace_bytes) {
        state.workspace = at_npu::native::empty_with_format(
            {static_cast<int64_t>(workspace_bytes)}, x.options().dtype(at::kByte), ACL_FORMAT_ND);
    }
    const auto workspace = state.workspace;
    auto* symmetric_ptr = static_cast<uint8_t*>(get_catccos_symm_ptr(symmetric_capacity));
    const uint64_t ffts = get_catccos_ffts();
    for (const auto& input : inputs) {
        c10_npu::NPUCachingAllocator::recordStream(input.tensor.storage().data_ptr(), npu_stream);
    }
    c10_npu::NPUCachingAllocator::recordStream(output.storage().data_ptr(), npu_stream);
    c10_npu::NPUCachingAllocator::recordStream(workspace.storage().data_ptr(), npu_stream);

    // Capture tensors by value until host submission. recordStream keeps
    // their storage live until the asynchronous device launch finishes.
    state.lastLaunchStream = npu_stream;
    at_npu::native::OpCommand::RunOpApi("catccos_ascend950_dispatch_ffn_combine", [=]() -> int {
        auto ptr = [](const at::Tensor& tensor) { return static_cast<uint8_t*>(tensor.data_ptr()); };
        CatccosKernel::catccos_ascend950_dispatch_ffn_combine_wrapper(
            block_dim, stream, ffts, ptr(x), ptr(expert_idx), ptr(gate_weight), ptr(w1), ptr(w1_scale), ptr(w2),
            ptr(w2_scale), ptr(output), ptr(workspace), symmetric_ptr, m, k, n, local_experts, top_k, rank_size);
        return 0;
    });
    return output;
}
#endif

int64_t finalize()
{
    auto& state = get_catccos_state();
    if (!state.initialized) {
        return 0;
    }

#ifdef CATCCOS_TORCH_A5
    c10_npu::NPUGuard guard(*state.device);
    if (state.lastLaunchStream) {
        synchronize_catccos_stream(*state.lastLaunchStream);
    }
#endif
    if (state.symmPtr != nullptr) {
#ifndef CATCCOS_TORCH_A5
        aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
        if (stream != nullptr) {
            auto sync_ret = aclrtSynchronizeStream(stream);
            TORCH_CHECK(sync_ret == ACL_SUCCESS, "aclrtSynchronizeStream before shmem_free failed, ret=", sync_ret);
        }
#endif
        shmem_free(state.symmPtr);
        state.symmPtr = nullptr;
        state.symmBytes = 0;
    }

#ifdef CATCCOS_TORCH_A5
    state.workspace = at::Tensor();
    state.device.reset();
    state.lastLaunchStream.reset();
#endif
    int64_t status = aclshmem_finalize();
    state.initialized = false;
    state.rankId = -1;
    state.rankSize = 0;
    TORCH_CHECK(status == ACLSHMEM_SUCCESS, "aclshmem_finalize failed: ", status);
    return 0;
}

} // namespace catccos

TORCH_LIBRARY(catccos, m)
{
    m.def("init(int rank_id, int rank_size, int local_mem_size, str ip_port) -> int");
#ifndef CATCCOS_TORCH_A5
    m.def("allgather_matmul(Tensor a, Tensor b, int rank_size) -> Tensor");
#else
    m.def("ascend950_dispatch_ffn_combine(Tensor x, Tensor expert_idx, Tensor gate_weight, "
          "Tensor w1, Tensor w1_scale, Tensor w2, Tensor w2_scale) -> Tensor");
#endif
    m.def("finalize() -> int");

    m.impl("init", &catccos::init);
#ifndef CATCCOS_TORCH_A5
    m.impl("allgather_matmul", torch::kPrivateUse1, &catccos::allgather_matmul);
#else
    m.impl("ascend950_dispatch_ffn_combine", torch::kPrivateUse1, &catccos::ascend950_dispatch_ffn_combine);
#endif
    m.impl("finalize", &catccos::finalize);
}
