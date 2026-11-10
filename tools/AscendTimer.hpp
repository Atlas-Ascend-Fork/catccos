
#ifndef ASCEND_TIMER_COMMON_H
#define ASCEND_TIMER_COMMON_H

#include <cstdint>
#include "info.h"
// ============================================================================
//  AscendTimer 用户配置区
//  ──────────────────────────────────────────────────────────────────────────
//  本文件是打点工具的唯一配置入口。
//  添加新的计时项只需编辑下方的列表，无需修改其他任何文件。
//
//  【格式】在列表末尾加一行:    X(YOUR_NAME)
//  【效果】enum 值 + CSV 名称映射自动生成，kernel 和 host 端同步生效
//  【注意】名称仅限 字母/数字/下划线，不要加逗号或分号
//
//  示例：要新增一个 AIV_COMM 阶段，只需在动态列表加一行 X(AIV_COMM)
//        即可在 kernel 中 timer.Tik(AIV_COMM) / timer.Tok(AIV_COMM)
//        CSV 输出中自动出现 AIV_COMM_0, AIV_COMM_1, ...
// ============================================================================

namespace AscendTimer {

// ======================== 固定计时项（整轮只记录一次） ========================
//   典型用途：整个 Kernel 的总耗时
//   添加格式：在列表末尾加一行  X(YOUR_NAME)
#define ASCEND_TIMER_FIXED_LIST \
    X(KERNEL_TIMING_IDX)        \
    // 继续添加

enum class FixedTiming : int {
#define X(name) name,
    ASCEND_TIMER_FIXED_LIST
#undef X
        FIXED_TIMING_COUNT // 哨兵，勿删
};

#define X(name) name = FixedTiming::name,
inline constexpr FixedTiming ASCEND_TIMER_FIXED_LIST FIXED_TIMING_ALIAS_COUNT = FixedTiming::FIXED_TIMING_COUNT;
#undef X
inline constexpr int FIXED_TIMING_COUNT = static_cast<int>(FixedTiming::FIXED_TIMING_COUNT);

// ======================== 动态计时项（每轮迭代可多次 Tik/Tok） ========================
//   典型用途：AIC 计算、AIV 通信、ReduceScatter 等可重复阶段
//   添加格式：在列表末尾加一行  X(YOUR_NAME)
//
//   已有示例：
//     X(AIC)     → kernel 中 timer.Tik(AIC) / timer.Tok(AIC)，CSV 列名 AIC_0, AIC_1, ...
//     X(AIV_RS)  → kernel 中 timer.Tik(AIV_RS) / timer.Tok(AIV_RS)，CSV 列名 AIV_RS_0, ...
#define ASCEND_TIMER_DYNAMIC_LIST \
    X(AIC)                        \
    X(AIV)                        \
    X(AIV_RS)                     \
    X(AIV_AG)                     \
    X(AIV_LOCAL_COPY)             \
    X(AIC_WAIT)                   \
    X(AIV_WAIT)

enum class DynamicTimingType : int {
#define X(name) name,
    ASCEND_TIMER_DYNAMIC_LIST
#undef X
        DYNAMIC_TYPE_COUNT_ENUM // 哨兵，勿删
};

#define X(name) name = DynamicTimingType::name,
inline constexpr DynamicTimingType ASCEND_TIMER_DYNAMIC_LIST DYNAMIC_TIMING_ALIAS_COUNT =
    DynamicTimingType::DYNAMIC_TYPE_COUNT_ENUM;
#undef X
inline constexpr int DYNAMIC_TYPE_COUNT_ENUM = static_cast<int>(DynamicTimingType::DYNAMIC_TYPE_COUNT_ENUM);

// ============================================================================
//  以下为内部常量，无需修改
// ============================================================================

// 所有动态项默认最大迭代次数
// 注意：存储 start/end 分开，因此每个事件需要 2 个存储 slot
#ifdef ASCEND_TIMER_MAX_DYNAMIC_ITER
inline constexpr int MAX_DYNAMIC_ITER = ASCEND_TIMER_MAX_DYNAMIC_ITER;
#else
inline constexpr int MAX_DYNAMIC_ITER = 150;
#endif
static_assert(MAX_DYNAMIC_ITER > 0 && MAX_DYNAMIC_ITER <= 32767, "timer count must fit int16_t");

// CPU端：对齐相关常量（128字节(16元素)对齐，避免多核写一个cache line）
// 每核原始计数器数 = 固定项*2(START/END) + 动态类型*最大迭代*2(START/END)
inline constexpr int N_TIMING_COUNTER_PER_CORE =
    FIXED_TIMING_COUNT * 2 + DYNAMIC_TYPE_COUNT_ENUM * MAX_DYNAMIC_ITER * 2;
inline constexpr int N_TIMING_COUNTER_PER_CORE_ALIGN = (N_TIMING_COUNTER_PER_CORE + 15) / 16 * 16;
inline constexpr int N_CORE_COUNT = BLOCK_NUM * 3; // GetCoreNum 不可用时的兼容 fallback
inline constexpr int N_TIMING_COUNTER = N_TIMING_COUNTER_PER_CORE_ALIGN * N_CORE_COUNT;

inline int GetTimerCoreCount(int platformCoreNum)
{
    int blockNum = platformCoreNum > 0 ? platformCoreNum : static_cast<int>(BLOCK_NUM);
    return blockNum * 3;
}

inline int GetTimingCounterSize(int platformCoreNum)
{
    return N_TIMING_COUNTER_PER_CORE_ALIGN * GetTimerCoreCount(platformCoreNum);
}

// 动态迭代次数存储区对齐配置
inline constexpr int DYNAMIC_TYPE_COUNT = DYNAMIC_TYPE_COUNT_ENUM;
inline constexpr int DYNAMIC_ITER_PER_CORE_ALIGN = (DYNAMIC_TYPE_COUNT + 15) / 16 * 16;
inline constexpr int DYNAMIC_ITER_TOTAL_SIZE = DYNAMIC_ITER_PER_CORE_ALIGN * N_CORE_COUNT;

inline int GetDynamicIterTotalSize(int platformCoreNum)
{
    return DYNAMIC_ITER_PER_CORE_ALIGN * GetTimerCoreCount(platformCoreNum);
}

inline int GetTotalBufferSize(int platformCoreNum)
{
    return GetTimingCounterSize(platformCoreNum) + GetDynamicIterTotalSize(platformCoreNum);
}

} // namespace AscendTimer

// ============================================================================
//  计时开关宏
// ============================================================================
#ifdef ENABLE_TIMER
#define TIMER_BLOCK(code) \
    do {                  \
        code;             \
    } while (0)
#else
#define TIMER_BLOCK(code) \
    do {                  \
    } while (0)
#endif

#endif // ASCEND_TIMER_COMMON_H
