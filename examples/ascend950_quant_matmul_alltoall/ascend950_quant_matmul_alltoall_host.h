
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND950_QUANT_MATMUL_ALLTOALL_HOST_H
#define ASCEND950_QUANT_MATMUL_ALLTOALL_HOST_H

#include "operator_registry.h"
#include "catlass/detail/alignment.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

// ============================================================================
// host 侧:设备内存分配与数据灌入、tiling 校验(评审与增量设计 §8)。
// 符号:m=每卡行数,N=每卡全列数(全局 M = P*m 自动整除),Y=[P*m, N/P] bf16。
// fmt:0 = MXFP8,1 = MXFP4(FP4 物理字节 = 逻辑元素数 / 2),
// 2 = BF16(BF16 物理字节 = 逻辑元素数 * 2,无 MX scale)。
// ============================================================================

// 扩展的 tiling 参数:在基线 CocTilingParams(m/N/K/m0/n0/k0/commInterval)
// 之上追加本算子的四项配置。
struct QuantMatmulAllToAllTilingParams : CocTilingParams {
    uint32_t fmt = 0;           // 0=MXFP8, 1=MXFP4, 2=BF16
    uint32_t slotNum = 2;       // 兼容旧命令行；kernel 按轮数分配完整 TX 区
    uint32_t laneNum = 1;       // 每 peer 的数据 QP 数(行条带并行度)
    uint32_t ranksPerNode = 16; // node-major 编址下的每节点 rank 数
};

class Ascend950QuantMatmulAllToAllOperator : public CatccosOperator {
public:
    static constexpr size_t STAGING_BYTES = 8 * 1024 * 1024;

    void AllocateDeviceSpace(
        KernelParams& params, const CocTilingParams& cocTiling, uint32_t rankId, std::string dataFile) override
    {
        AllocateDeviceSpace(params, ToQuantTilingParams(cocTiling), rankId, std::move(dataFile));
    }

    /// 为本 rank 分配并灌入输入:三个 fmt 的字节折算不同
    /// (MXFP8 1B/元素、MXFP4 每字节打包 2 元素、BF16 2B/元素且无 scale)。
    /// 多卡(P>1)时 Y 不在这里分配——main.cpp 把它重定向进对称区,
    /// 使本卡 AIV 可以对端 UDMA/RoCE PUT 写入本卡输出。
    void AllocateDeviceSpace(
        KernelParams& params, const QuantMatmulAllToAllTilingParams& cocTiling, uint32_t rankId, std::string dataFile)
    {
        const uint32_t m = cocTiling.m;
        const uint32_t n = cocTiling.n;
        const uint32_t k = cocTiling.k;
        const uint32_t p = cocTiling.rankSize;
        const uint32_t q = n / p;
        const bool isFp4 = (cocTiling.fmt == 1); // FP4:每字节打包 2 元素
        const bool isBf16 = (cocTiling.fmt == 2);
        const uint32_t pack = isFp4 ? 2 : 1;
        const uint32_t elementBytes = isBf16 ? sizeof(bfloat16_t) : 1;

        const size_t lenA = static_cast<size_t>(m) * k;
        const size_t lenB = static_cast<size_t>(k) * n;
        constexpr uint32_t MX_GROUP = 32;
        // 物理布局为 [·, ceil(K/64), 2] 双打包:组数上取整到偶数,防 K%64==32 时
        // 消费端按物理槽读取越界(原评审 §8.2 的告诫)
        const size_t mxScaleK = (CeilDiv<MX_GROUP>(k) + 1) / 2 * 2;
        const size_t lenAScale = static_cast<size_t>(m) * mxScaleK;
        const size_t lenBScale = static_cast<size_t>(n) * mxScaleK;

        const size_t aSize = lenA / pack * elementBytes;
        const size_t bSize = lenB / pack * elementBytes;
        const size_t aScaleSize = lenAScale; // E8M0 1B
        const size_t bScaleSize = lenBScale;

        // A: [m, K] 量化数据(各卡不同)
        uint8_t* aDevice = nullptr;
        CheckAcl(aclrtMalloc((void**)(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST), "allocate A");
        FillOrLoad(aDevice, aSize, dataFile + "/rank_" + std::to_string(rankId) + "_a.bin", isBf16);

        // B(weight): [K, N] ND 行主序,各卡相同(决议:不转 NZ、不置尾)
        uint8_t* bDevice = nullptr;
        CheckAcl(aclrtMalloc((void**)(&bDevice), bSize, ACL_MEM_MALLOC_HUGE_FIRST), "allocate B");
        FillOrLoad(bDevice, bSize, dataFile + "/rank_" + std::to_string(rankId) + "_b.bin", isBf16);

        // Y 始终绑定到 main.cpp 分配的对称区,供本卡及对端 PUT 直写。
        uint8_t* yDevice = nullptr;

        uint8_t* aScaleDevice = nullptr;
        uint8_t* bScaleDevice = nullptr;
        if (!isBf16) {
            // A scale: [m, ceil(K/32)] E8M0
            CheckAcl(aclrtMalloc((void**)(&aScaleDevice), aScaleSize, ACL_MEM_MALLOC_HUGE_FIRST), "allocate A scale");
            FillOrLoad(aScaleDevice, aScaleSize, dataFile + "/rank_" + std::to_string(rankId) + "_a_scale.bin");

            // B scale: [N, ceil(K/32)] E8M0(MakeMxScaleLayout<...,true> 的转置布局,与基线一致)
            CheckAcl(aclrtMalloc((void**)(&bScaleDevice), bScaleSize, ACL_MEM_MALLOC_HUGE_FIRST), "allocate B scale");
            FillOrLoad(bScaleDevice, bScaleSize, dataFile + "/rank_" + std::to_string(rankId) + "_b_scale.bin");
        }

        params.SetKernelParams(aDevice, bDevice, yDevice, aScaleDevice, bScaleDevice);
    }

    /// 把 Y(多卡时在对称区)拷回 host 并写出 output_<rank>.bin,
    /// 供 verify_result.py 按 FP32 golden 逐元素校验。
    void WriteResultFile(
        const KernelParams& params, const CocTilingParams& cocTiling, uint32_t rankId, std::string dataFile) override
    {
        const size_t lenY = static_cast<size_t>(cocTiling.m) * cocTiling.n; // [P*m, q] 元素数 = m*N
        const size_t ySize = lenY * sizeof(bfloat16_t);
        WriteDeviceFile(dataFile + "/output_" + std::to_string(rankId) + ".bin", params.ptrC, ySize);
    }

    size_t GetWorkspaceSize(const CocTilingParams& cocTiling) override { return 0; }

    CocCommType GetActualKernelType(const CocTilingParams& cocTiling) override
    {
        return CocCommType::ASCEND950_QUANT_MATMUL_ALLTOALL;
    }

    /// 本算子专属校验:每个 peer 的每个通信窗口必须小于单 QP 请求上限。
    bool ValidateQuantTilingParams(uint32_t rankSize, const QuantMatmulAllToAllTilingParams& cocTiling)
    {
        if (!ValidateCocTilingParams(rankSize, cocTiling)) {
            return false;
        }

        const uint64_t m = cocTiling.m;
        const uint64_t n = cocTiling.n;
        const uint64_t p = rankSize;
        const uint64_t q = n / p;
        if (cocTiling.fmt > 2 || cocTiling.laneNum == 0 || cocTiling.laneNum > 4) {
            return false;
        }
        // HierarchicalCommSwizzle supports up to 32 peers and 16 ranks per UDMA domain.
        if (cocTiling.ranksPerNode == 0 || cocTiling.ranksPerNode > 16 || p > 32) {
            return false;
        }

        const uint64_t h = static_cast<uint64_t>(cocTiling.commInterval) * cocTiling.m0;
        const uint64_t stripeRows = h / cocTiling.laneNum + (h % cocTiling.laneNum != 0);
        return stripeRows <= (256ULL << 20) / (q * 2);
    }

    // 参数校验(评审 §8:统一契约 + uint64 溢出检查)
    bool CheckCocTilingParams(uint32_t rankSize, const CocTilingParams& cocTiling) override
    {
        const uint64_t m = cocTiling.m;
        const uint64_t n = cocTiling.n;
        const uint64_t k = cocTiling.k;
        const uint64_t p = rankSize;
        if (m == 0 || n == 0 || k == 0 || p < 2 || p > 32 || cocTiling.rankSize != p) {
            return false;
        }
        const uint64_t q = n / p;
        if (n % p != 0) {
            return false; // 切 N 等长交换(评审 §8.4)
        }
        if (cocTiling.m0 == 0 || cocTiling.n0 == 0 || q % cocTiling.n0 != 0) {
            return false; // tile 不跨 rank 边界(保持整 tile 路由,评审 §8.4)
        }
        if (k % 32 != 0) {
            return false; // K 为 32 的倍数(决议;kernel 内按 64 补齐)
        }
        if (cocTiling.commInterval == 0) {
            return false;
        }
        // uint64 溢出与单请求上限(评审 §8.5/§5.2):Y 总量、条带字节、QP 数
        if (m > UINT32_MAX / p || m > ((1ULL << 40) - 1) / (n * 2)) {
            return false;
        }
        const uint64_t h = static_cast<uint64_t>(cocTiling.commInterval) * cocTiling.m0;
        if (h > UINT32_MAX / p) {
            return false;
        }
        return true;
    }

private:
    static QuantMatmulAllToAllTilingParams ToQuantTilingParams(const CocTilingParams& cocTiling)
    {
        QuantMatmulAllToAllTilingParams quantTiling;
        static_cast<CocTilingParams&>(quantTiling) = cocTiling;
        return quantTiling;
    }

    // 优先从数据文件加载;dataFile 为空时填 1(与基线示例一致)
    static void FillOrLoad(uint8_t* device, size_t size, const std::string& file, bool bf16Ones = false)
    {
        if (!file.empty() && CheckFileExist(file)) {
            ReadDeviceFile(file, device, size);
            return;
        }
        FillConstant(device, size, bf16Ones);
    }

    static void CheckAcl(aclError status, const char* operation)
    {
        if (status != ACL_SUCCESS) {
            throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status));
        }
    }

    /// Large shapes used to allocate one full pinned buffer per input and one
    /// for the output in every rank process. Sixteen concurrent ranks could
    /// exhaust host memory before any missing output was reported. Keep only a
    /// bounded staging window; the device allocation still contains the full
    /// tensor.
    static void ReadDeviceFile(const std::string& file, uint8_t* device, size_t bytes)
    {
        std::ifstream input(file, std::ios::binary | std::ios::ate);
        if (!input) {
            throw std::runtime_error("cannot open input: " + file);
        }
        const std::streamoff fileBytes = input.tellg();
        if (fileBytes < 0 || static_cast<uint64_t>(fileBytes) != bytes) {
            throw std::runtime_error(
                "wrong input size: " + file + " expected=" + std::to_string(bytes) +
                " actual=" + std::to_string(fileBytes));
        }
        input.seekg(0);

        const size_t stagingBytes = std::min(bytes, STAGING_BYTES);
        void* host = nullptr;
        CheckAcl(aclrtMallocHost(&host, stagingBytes), "allocate input staging buffer");
        auto* staging = static_cast<uint8_t*>(host);
        for (size_t offset = 0; offset < bytes; offset += stagingBytes) {
            const size_t chunk = std::min(stagingBytes, bytes - offset);
            if (!input.read(reinterpret_cast<char*>(staging), chunk)) {
                CheckAcl(aclrtFreeHost(host), "free input staging buffer");
                throw std::runtime_error("failed to read input: " + file);
            }
            CheckAcl(aclrtMemcpy(device + offset, chunk, staging, chunk, ACL_MEMCPY_HOST_TO_DEVICE), "input H2D copy");
        }
        CheckAcl(aclrtFreeHost(host), "free input staging buffer");
    }

    static void WriteDeviceFile(const std::string& file, const uint8_t* device, size_t bytes)
    {
        const std::string temporary = file + ".tmp";
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("cannot create output: " + temporary);
        }

        const size_t stagingBytes = std::min(bytes, STAGING_BYTES);
        void* host = nullptr;
        CheckAcl(aclrtMallocHost(&host, stagingBytes), "allocate output staging buffer");
        auto* staging = static_cast<uint8_t*>(host);
        for (size_t offset = 0; offset < bytes; offset += stagingBytes) {
            const size_t chunk = std::min(stagingBytes, bytes - offset);
            CheckAcl(aclrtMemcpy(staging, chunk, device + offset, chunk, ACL_MEMCPY_DEVICE_TO_HOST), "output D2H copy");
            if (!output.write(reinterpret_cast<const char*>(staging), chunk)) {
                CheckAcl(aclrtFreeHost(host), "free output staging buffer");
                output.close();
                std::remove(temporary.c_str());
                throw std::runtime_error("failed to write output: " + temporary);
            }
        }
        CheckAcl(aclrtFreeHost(host), "free output staging buffer");
        output.close();
        if (!output) {
            std::remove(temporary.c_str());
            throw std::runtime_error("failed to flush output: " + temporary);
        }
        std::remove(file.c_str());
        if (std::rename(temporary.c_str(), file.c_str()) != 0) {
            std::remove(temporary.c_str());
            throw std::runtime_error("failed to publish output: " + file);
        }
    }

    static void FillConstant(uint8_t* device, size_t bytes, bool bf16Ones)
    {
        const size_t stagingBytes = std::min(bytes, STAGING_BYTES);
        std::vector<uint8_t> staging(stagingBytes, 1);
        if (bf16Ones) {
            // BF16 1.0 is 0x3f80. Avoid allocating the complete tensor in
            // pinned host memory for the no-data fallback.
            for (size_t offset = 0; offset + sizeof(uint16_t) <= staging.size(); offset += sizeof(uint16_t)) {
                staging[offset] = 0x80;
                staging[offset + 1] = 0x3f;
            }
        }
        for (size_t offset = 0; offset < bytes; offset += stagingBytes) {
            const size_t chunk = std::min(stagingBytes, bytes - offset);
            CheckAcl(
                aclrtMemcpy(device + offset, chunk, staging.data(), chunk, ACL_MEMCPY_HOST_TO_DEVICE),
                "constant H2D copy");
        }
    }

    static bool CheckFileExist(const std::string& file)
    {
        std::ifstream f(file);
        return f.good();
    }
};

REGISTER_OPERATOR("Ascend950QuantMatmulAllToAll", Ascend950QuantMatmulAllToAllOperator);

#endif // ASCEND950_QUANT_MATMUL_ALLTOALL_HOST_H
