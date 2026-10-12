
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_TIMER_HOST_H
#define ASCEND_TIMER_HOST_H

#include <vector>
#include <iostream>
#include <unordered_map>
#include <fstream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <sstream>

#include "AscendTimer.hpp"
#include "acl/acl_rt.h"

#if __has_include("tiling/platform/platform_ascendc.h")
#ifdef ASCENDC_ASSERT
#undef ASCENDC_ASSERT
#endif
#include "tiling/platform/platform_ascendc.h"
#endif

class AscendTimerHost {
public:
    // ========== 常量定义 ==========
    static constexpr float kCycleToTimeBase = 50.0f;       // 周期转微秒的基准值
    static constexpr int kCoresPerGroup = 3;               // 每组的核数
    static constexpr int kSubGroupIdForAIC = 0;            // AIC 的子组 ID
    static inline int runtimePlatformCoreNum_ = BLOCK_NUM; // GetCoreNum() 返回值，等价于原 BLOCK_NUM

    // ========== 1. 总缓冲区大小计算 ==========
    static inline int GetPlatformCoreNum()
    {
#if __has_include("tiling/platform/platform_ascendc.h")
        auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        if (platform != nullptr) {
            int coreNum = static_cast<int>(platform->GetCoreNum());
            if (coreNum > 0)
                return coreNum;
        }
#endif
        return static_cast<int>(BLOCK_NUM);
    }

    static inline int GetPlatformCoreNumAic()
    {
#if __has_include("tiling/platform/platform_ascendc.h")
        auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        if (platform != nullptr) {
            int coreNum = static_cast<int>(platform->GetCoreNumAic());
            if (coreNum > 0)
                return coreNum;
        }
#endif
        return static_cast<int>(BLOCK_NUM);
    }

    static inline int GetPlatformCoreNumAiv()
    {
#if __has_include("tiling/platform/platform_ascendc.h")
        auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        if (platform != nullptr) {
            int coreNum = static_cast<int>(platform->GetCoreNumAiv());
            if (coreNum > 0)
                return coreNum;
        }
#endif
        return static_cast<int>(BLOCK_NUM);
    }

    static inline void ConfigureCoreNum(int coreNum)
    {
        runtimePlatformCoreNum_ = coreNum > 0 ? coreNum : static_cast<int>(BLOCK_NUM);
    }

    static inline int GetTimerCoreCount() { return AscendTimer::GetTimerCoreCount(runtimePlatformCoreNum_); }

    static inline int GetTimingCounterSize() { return AscendTimer::GetTimingCounterSize(runtimePlatformCoreNum_); }

    static inline int GetDynamicIterTotalSize()
    {
        return AscendTimer::GetDynamicIterTotalSize(runtimePlatformCoreNum_);
    }

    static inline int GetTotalTimerBufferSize() { return AscendTimer::GetTotalBufferSize(runtimePlatformCoreNum_); }

    // ========== 2. 名称映射表 ==========
    static inline const std::unordered_map<int, std::string> fixedNameMap = {
#define X(name) {static_cast<int>(AscendTimer::name), #name},
        ASCEND_TIMER_FIXED_LIST
#undef X
    };

    static inline const std::unordered_map<AscendTimer::DynamicTimingType, std::string> dynamicNameMap = {
#define X(name) {AscendTimer::name, #name},
        ASCEND_TIMER_DYNAMIC_LIST
#undef X
    };

    // ========== 3. 动态项名称解析 ==========
    static inline bool ParseDynamicName(const std::string& name, AscendTimer::DynamicTimingType& type, int& iter)
    {
        // 优先进行全名匹配
        for (const auto& pair : dynamicNameMap) {
            if (pair.second == name) {
                type = pair.first;
                iter = 0; // 累加模式的数据固定去读取 0 号槽位
                return true;
            }
        }

        size_t lastUnderscore = name.find_last_of('_');
        if (lastUnderscore == std::string::npos) {
            return false;
        }

        std::string prefix = name.substr(0, lastUnderscore);
        std::string iterStr = name.substr(lastUnderscore + 1);

        try {
            iter = std::stoi(iterStr);
        } catch (...) {
            return false;
        }

        // 验证前缀是否在映射表中
        for (const auto& pair : dynamicNameMap) {
            if (pair.second == prefix) {
                type = pair.first;
                return true;
            }
        }

        return false;
    }

    // ========== 4. 计时结果结构体 ==========
    struct TimerResult {
        int coreCount = AscendTimer::N_CORE_COUNT;             // 实际导出的 AIC+AIV 总核数
        int timingCounterSize = AscendTimer::N_TIMING_COUNTER; // baseCycleData 的实际大小
        int dynamicIterTotalSize = AscendTimer::DYNAMIC_ITER_TOTAL_SIZE;
        std::vector<int64_t> baseCycleData;            // 基础计时数据（N_TIMING_COUNTER）
        std::vector<int64_t> dynamicActualIterPerCore; // 每核动态迭代次数（DYNAMIC_ITER_TOTAL_SIZE）
        std::vector<int64_t> dynamicActualIterMax; // 每个动态项的全局最大迭代次数（DYNAMIC_TYPE_COUNT）
        std::vector<bool> isAccumulate;            // 新增：记录该类型是否为累加模式
    };

    inline static bool IsComplete(const TimerResult& result)
    {
        return result.coreCount > 0 && result.timingCounterSize > 0 && result.dynamicIterTotalSize >= 0 &&
               result.timingCounterSize == result.coreCount * AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN &&
               result.dynamicIterTotalSize == result.coreCount * AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN &&
               result.baseCycleData.size() == static_cast<size_t>(result.timingCounterSize) &&
               result.dynamicActualIterPerCore.size() == static_cast<size_t>(result.dynamicIterTotalSize) &&
               result.dynamicActualIterMax.size() == AscendTimer::DYNAMIC_TYPE_COUNT &&
               result.isAccumulate.size() == AscendTimer::DYNAMIC_TYPE_COUNT;
    }

    // ========== 5. 动态项索引计算 ==========
    static inline int GetDynamicTimingIdx(AscendTimer::DynamicTimingType type, int iter)
    {
        int targetIdx = AscendTimer::FIXED_TIMING_COUNT + static_cast<int>(type) * AscendTimer::MAX_DYNAMIC_ITER + iter;
        if (iter >= AscendTimer::MAX_DYNAMIC_ITER) {
            std::cerr << "[WARNING] 动态项 " << dynamicNameMap.at(type) << " 迭代次数 " << iter
                      << " 超出全局最大迭代数 " << AscendTimer::MAX_DYNAMIC_ITER << std::endl;
        }
        if (targetIdx >= AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN) {
            std::cerr << "[WARNING] 动态项 " << dynamicNameMap.at(type) << " 索引 " << targetIdx
                      << " 超出每核对齐后最大计数器数 " << AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN << std::endl;
        }
        return targetIdx;
    }

    // ========== 6. 计时项名称获取 ==========
    // 只返回实际有数据的计时项，Accumulate模式去除_0后缀
    static inline std::vector<std::string> GetTimingNames(const TimerResult& timerResult)
    {
        std::vector<std::string> names;
        const auto& fixedMap = fixedNameMap;

        // 固定项名称
        for (int i = 0; i < AscendTimer::FIXED_TIMING_COUNT; ++i) {
            auto it = fixedMap.find(i);
            names.push_back(it != fixedMap.end() ? it->second : "UNKNOWN_FIXED_" + std::to_string(i));
        }

        // 动态项名称：仅输出有实际迭代数据的类型
        for (int t = 0; t < AscendTimer::DYNAMIC_TYPE_COUNT; ++t) {
            auto type = static_cast<AscendTimer::DynamicTimingType>(t);
            if (!dynamicNameMap.count(type))
                continue;

            int64_t actualMaxIter =
                t < timerResult.dynamicActualIterMax.size() ? timerResult.dynamicActualIterMax[t] : 0;
            if (actualMaxIter <= 0)
                continue;

            const std::string& prefix = dynamicNameMap.at(type);

            // 🌟 判断：如果是 Accumulate 模式，只输出一个单纯的列名
            if (timerResult.isAccumulate.size() > t && timerResult.isAccumulate[t]) {
                names.push_back(prefix); // 输出如 "SYNPIC_TIME" (没有 _0)
            } else {
                for (int iter = 0; iter < actualMaxIter; ++iter) {
                    names.push_back(prefix + "_" + std::to_string(iter));
                }
            }
        }

        return names;
    }

    // ========== 7. 从Device拷贝计时结果 ==========
    static inline TimerResult GetTimeResult(void* deviceAddr)
    {
        TimerResult result;
        result.coreCount = GetTimerCoreCount();
        result.timingCounterSize = GetTimingCounterSize();
        result.dynamicIterTotalSize = GetDynamicIterTotalSize();
        int totalBufferSize = GetTotalTimerBufferSize();
        std::vector<int64_t> allData(totalBufferSize, 0);

        auto ret = aclrtMemcpy(
            allData.data(), totalBufferSize * sizeof(int64_t), deviceAddr, totalBufferSize * sizeof(int64_t),
            ACL_MEMCPY_DEVICE_TO_HOST);

        if (ret != ACL_SUCCESS) {
            std::cerr << "Memcpy device→host failed! ERROR: " << ret << std::endl;
            return result;
        }

        result.baseCycleData = std::vector<int64_t>(allData.begin(), allData.begin() + result.timingCounterSize);
        result.dynamicActualIterPerCore =
            std::vector<int64_t>(allData.begin() + result.timingCounterSize, allData.end());

        // 计算每个动态项的全局最大迭代次数
        result.dynamicActualIterMax.resize(AscendTimer::DYNAMIC_TYPE_COUNT, 0);
        result.isAccumulate.resize(AscendTimer::DYNAMIC_TYPE_COUNT, false); // 初始化

        for (int coreId = 0; coreId < result.coreCount; ++coreId) {
            int coreOffset = coreId * AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN;
            for (int type = 0; type < AscendTimer::DYNAMIC_TYPE_COUNT; ++type) {
                int idx = coreOffset + type;
                if (idx >= result.dynamicActualIterPerCore.size())
                    continue;

                int64_t val = result.dynamicActualIterPerCore[idx];

                // 🌟 接收到设备端发来的 Accumulate 负数暗号！
                if (val < 0) {
                    result.isAccumulate[type] = true;
                    val = -val;
                    result.dynamicActualIterPerCore[idx] = val; // 恢复为正数，供 IterCount 列输出
                }

                if (val > result.dynamicActualIterMax[type]) {
                    result.dynamicActualIterMax[type] = val;
                }
            }
        }

        return result;
    }

    // ========== 8. 周期数转时间（微秒） ==========
    inline static float CycleToTime(int64_t cycles) { return static_cast<float>(cycles) / kCycleToTimeBase; }

    // ========== 9. 写入原始时间戳数据（CSV格式，包含start/end cycle值）==========
    inline static bool WriteTimeResultsRaw(const std::string& filename, const TimerResult& timerResult)
    {
        if (!IsComplete(timerResult))
            return false;
        std::ofstream outfile(filename);
        if (!outfile.is_open())
            return false;

        const auto& timingNames = GetTimingNames(timerResult);
        const int totalItems = timingNames.size();
        const auto& cycleResult = timerResult.baseCycleData;
        const auto& dynamicMaxIter = timerResult.dynamicActualIterMax;

        outfile << "group_id,core_type,sub_group_id";
        for (const auto& name : timingNames) {
            outfile << "," << name << "_start_cycle," << name << "_end_cycle," << name << "_duration_us";
        }
        // _IterCount 列也只输出有数据的类型
        for (int t = 0; t < AscendTimer::DYNAMIC_TYPE_COUNT; ++t) {
            if (t < dynamicMaxIter.size() && dynamicMaxIter[t] <= 0)
                continue;
            auto type = static_cast<AscendTimer::DynamicTimingType>(t);
            if (dynamicNameMap.count(type))
                outfile << "," << dynamicNameMap.at(type) << "_IterCount";
        }
        outfile << std::endl;

        for (int i = 0; i < timerResult.coreCount; i++) {
            int groupId = i / kCoresPerGroup;
            int subGroupId = i % kCoresPerGroup;
            std::string coreType = (subGroupId == kSubGroupIdForAIC) ? "AIC" : "AIV";
            int subId = (coreType == "AIC") ? 0 : (subGroupId - 1);

            int baseIndex = groupId * AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN * kCoresPerGroup;
            if (coreType == "AIV")
                baseIndex += AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN * subGroupId;

            outfile << groupId << "," << coreType << "," << subId;

            for (int nameIdx = 0; nameIdx < totalItems; ++nameIdx) {
                const std::string& currName = timingNames[nameIdx];
                int elementIdx = -1;

                if (nameIdx < AscendTimer::FIXED_TIMING_COUNT) {
                    elementIdx = baseIndex + nameIdx * 2;
                } else {
                    AscendTimer::DynamicTimingType dynType;
                    int iter;
                    if (ParseDynamicName(currName, dynType, iter)) {
                        int dynMemIdx = GetDynamicTimingIdx(dynType, iter);
                        elementIdx = baseIndex + dynMemIdx * 2;
                    }
                }

                // 越界检查
                if (elementIdx < 0 || elementIdx + 1 >= timerResult.timingCounterSize) {
                    outfile << ",0,0,0";
                } else {
                    int64_t start_cycles = cycleResult[elementIdx];
                    int64_t end_cycles = cycleResult[elementIdx + 1];
                    // 如果是0基准(Accumulate模式)，耗时就是end_cycles，否则正常相减
                    float duration_us = 0.0f;
                    if (start_cycles == 0) {
                        duration_us = CycleToTime(end_cycles);
                    } else if (end_cycles > start_cycles) {
                        duration_us = CycleToTime(end_cycles - start_cycles);
                    }
                    outfile << "," << start_cycles << "," << end_cycles << "," << duration_us;
                }
            }

            for (int t = 0; t < AscendTimer::DYNAMIC_TYPE_COUNT; ++t) {
                if (t < dynamicMaxIter.size() && dynamicMaxIter[t] <= 0)
                    continue;
                int idxIter = i * AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN + t;
                outfile << ","
                        << (idxIter < timerResult.dynamicActualIterPerCore.size() ?
                                timerResult.dynamicActualIterPerCore[idxIter] :
                                0);
            }
            outfile << std::endl;
        }
        outfile.close();
        return true;
    }

    // ========== 10. 写入Duration数据 ==========
    inline static bool WriteTimeResultsDuration(const std::string& filename, const TimerResult& timerResult)
    {
        if (!IsComplete(timerResult))
            return false;
        std::ofstream outfile(filename);
        if (!outfile.is_open())
            return false;

        const auto& timingNames = GetTimingNames(timerResult);
        const int totalItems = timingNames.size();
        const auto& cycleResult = timerResult.baseCycleData;
        const auto& dynamicMaxIter = timerResult.dynamicActualIterMax;

        outfile << "group_id,core_type,sub_group_id";
        for (const auto& name : timingNames) {
            outfile << "," << name << "_duration(us)";
        }
        // _IterCount 列也只输出有数据的类型
        for (int t = 0; t < AscendTimer::DYNAMIC_TYPE_COUNT; ++t) {
            if (t < dynamicMaxIter.size() && dynamicMaxIter[t] <= 0)
                continue;
            auto type = static_cast<AscendTimer::DynamicTimingType>(t);
            if (dynamicNameMap.count(type))
                outfile << "," << dynamicNameMap.at(type) << "_IterCount";
        }
        outfile << std::endl;

        for (int i = 0; i < timerResult.coreCount; i++) {
            int groupId = i / kCoresPerGroup;
            int subGroupId = i % kCoresPerGroup;
            std::string coreType = (subGroupId == kSubGroupIdForAIC) ? "AIC" : "AIV";
            int subId = (coreType == "AIC") ? 0 : (subGroupId - 1);

            int baseIndex = groupId * AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN * kCoresPerGroup;
            if (coreType == "AIV")
                baseIndex += AscendTimer::N_TIMING_COUNTER_PER_CORE_ALIGN * subGroupId;

            outfile << groupId << "," << coreType << "," << subId;

            for (int nameIdx = 0; nameIdx < totalItems; ++nameIdx) {
                const std::string& currName = timingNames[nameIdx];
                int elementIdx = -1;

                if (nameIdx < AscendTimer::FIXED_TIMING_COUNT) {
                    elementIdx = baseIndex + nameIdx * 2;
                } else {
                    AscendTimer::DynamicTimingType dynType;
                    int iter;
                    if (ParseDynamicName(currName, dynType, iter)) {
                        int dynMemIdx = GetDynamicTimingIdx(dynType, iter);
                        elementIdx = baseIndex + dynMemIdx * 2;
                    }
                }

                if (elementIdx < 0 || elementIdx + 1 >= timerResult.timingCounterSize) {
                    outfile << ",0";
                } else {
                    int64_t start_cycles = cycleResult[elementIdx];
                    int64_t end_cycles = cycleResult[elementIdx + 1];
                    float duration_us = 0.0f;
                    if (start_cycles == 0) {
                        duration_us = CycleToTime(end_cycles);
                    } else if (end_cycles > start_cycles) {
                        duration_us = CycleToTime(end_cycles - start_cycles);
                    }
                    outfile << "," << duration_us;
                }
            }

            for (int t = 0; t < AscendTimer::DYNAMIC_TYPE_COUNT; ++t) {
                if (t < dynamicMaxIter.size() && dynamicMaxIter[t] <= 0)
                    continue;
                int idxIter = i * AscendTimer::DYNAMIC_ITER_PER_CORE_ALIGN + t;
                outfile << ","
                        << (idxIter < timerResult.dynamicActualIterPerCore.size() ?
                                timerResult.dynamicActualIterPerCore[idxIter] :
                                0);
            }
            outfile << std::endl;
        }
        outfile.close();
        return true;
    }

    inline static bool WriteAllTimerResults(
        const std::string& rawFilename, const std::string& durationFilename, const TimerResult& timerResult)
    {
        if (!IsComplete(timerResult)) {
            std::cerr << "Incomplete timer result; CSV export skipped" << std::endl;
            return false;
        }
        bool rawOk = WriteTimeResultsRaw(rawFilename, timerResult);
        bool durOk = WriteTimeResultsDuration(durationFilename, timerResult);
        return rawOk && durOk;
    }
};

#endif // ASCEND_TIMER_HOST_H
