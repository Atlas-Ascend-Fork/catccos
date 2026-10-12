
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_DGEMM_DEVICE_DEVICE_DGEMM_HPP
#define CATCCOS_DGEMM_DEVICE_DEVICE_DGEMM_HPP

#include <cstddef>
#include <acl/acl.h>
#include "catlass/catlass.hpp"
#include "catlass/status.hpp"
#include "kernel_adapter.hpp"
#include "AscendTimer_host.hpp"

namespace Catccos::DGemm::Device {

template <class DGemmKernel>
class DeviceDGemm {
public:
    using Kernel = DGemmKernel;
    using Arguments = typename DGemmKernel::Arguments;
    using Params = typename DGemmKernel::Params;

private:
    Params params_;
    uint8_t* timerBuffer_ = nullptr;
    size_t timerBufferSize_ = 0;

public:
    DeviceDGemm() {}
    ~DeviceDGemm()
    {
        if (timerBuffer_ != nullptr) {
            aclrtFree(timerBuffer_);
            timerBuffer_ = nullptr;
            timerBufferSize_ = 0;
        }
    }

    Params const& params() const { return params_; }

    static size_t GetWorkspaceSize(Arguments const& args) { return DGemmKernel::GetWorkspaceSize(args); }

    Catlass::Status Initialize(Arguments const& args, uint8_t* workspace = nullptr)
    {
        params_ = DGemmKernel::ToUnderlyingArguments(args, workspace);
        return Catlass::Status::kSuccess;
    }

    void ExportTimerCsv()
    {
#ifdef ENABLE_TIMER
        if (timerBuffer_ == nullptr) {
            fprintf(stderr, "timerBuffer_ is nullptr, returning\n");
            return;
        }

        int rankIdInt = shmem_my_pe();
        uint32_t rankId = (rankIdInt >= 0) ? static_cast<uint32_t>(rankIdInt) : 0;

        size_t timerBufferSize = AscendTimerHost::GetTotalTimerBufferSize() * sizeof(int64_t);
        if (timerBufferSize == 0)
            return;

        std::string out_dir = "output_timer";
        mkdir(out_dir.c_str(), 0777);
        std::string csv_path = out_dir + "/timer_rank_" + std::to_string(rankId) + ".csv";

        auto time_result = AscendTimerHost::GetTimeResult(timerBuffer_);
        std::string data_dir = out_dir + "/data";
        mkdir(data_dir.c_str(), 0777);
        std::string raw_csv_path = data_dir + "/timer_rank_" + std::to_string(rankId) + "_timeline.csv";
        std::string dur_csv_path = out_dir + "/timer_rank_" + std::to_string(rankId) + ".csv";
        if (!AscendTimerHost::WriteAllTimerResults(raw_csv_path, dur_csv_path, time_result)) {
            fprintf(stderr, "[Timer] CSV export failed\n");
            return;
        }
        fprintf(stderr, "[Timer] Raw CSV: %s\n", raw_csv_path.c_str());
        fprintf(stderr, "[Timer] Duration CSV: %s\n", dur_csv_path.c_str());
#endif
    }

    inline Catlass::Status Run(aclrtStream stream, uint32_t blockDim, uint64_t fftsAddr)
    {
#ifdef ENABLE_TIMER
        (void)AscendTimerHost::GetPlatformCoreNum();
        AscendTimerHost::ConfigureCoreNum(static_cast<int>(blockDim));
        size_t timerBufferSize = AscendTimerHost::GetTotalTimerBufferSize() * sizeof(int64_t);

        if (timerBufferSize > timerBufferSize_ && timerBuffer_ != nullptr) {
            aclrtFree(timerBuffer_);
            timerBuffer_ = nullptr;
            timerBufferSize_ = 0;
        }

        if (timerBufferSize > 0 && timerBuffer_ == nullptr) {
            ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&timerBuffer_), timerBufferSize, ACL_MEM_MALLOC_HUGE_FIRST));
            timerBufferSize_ = timerBufferSize;
        }
#endif

        if (fftsAddr == 0) {
            KernelAdapter<DGemmKernel><<<blockDim, nullptr, stream>>>(params_, timerBuffer_);
        } else {
            KernelAdapter<DGemmKernel><<<blockDim, nullptr, stream>>>(params_, fftsAddr, timerBuffer_);
        }

        ACL_CHECK(aclrtSynchronizeStream(stream));
#ifdef ENABLE_TIMER
        ExportTimerCsv();
#endif

        return Catlass::Status::kSuccess;
    }

    inline Catlass::Status operator()(aclrtStream stream, uint32_t blockDim, uint64_t fftsAddr)
    {
        return Run(stream, blockDim, fftsAddr);
    }
};

} // namespace Catccos::DGemm::Device

#endif // CATCCOS_DGEMM_DEVICE_DEVICE_DGEMM_HPP
