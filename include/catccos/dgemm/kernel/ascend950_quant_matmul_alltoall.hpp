/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CATCCOS_DGEMM_KERNEL_ASCEND950_QUANT_MATMUL_ALLTOALL_HPP
#define CATCCOS_DGEMM_KERNEL_ASCEND950_QUANT_MATMUL_ALLTOALL_HPP

#include <cstdint>
#include <type_traits>

#include "catccos/catccos.hpp"
#include "catccos/comm/tile/tile_hierarchical_ready.hpp"
#include "catccos/detail/remote_copy_type.hpp"

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/matrix_coord.hpp"
#include "tla/tensor.hpp"

// CANN 9.3's platform header selects its host assertion branch for Bisheng
// device compilation because --cce-aicore-arch defines __CCE_AICORE__ rather
// than __NPU_DEVICE__.  SHMEM v1.8.0 uses ASCENDC_ASSERT in device headers,
// where the host branch would call the unavailable host raise() path.  Keep
// the compatibility local to the QP users so public headers retain stock
// semantics.
#if defined(__CCE_AICORE__)
#include "tiling/platform/platform_ascendc.h"
#ifdef ASCENDC_ASSERT
#undef ASCENDC_ASSERT
#endif
#define ASCENDC_ASSERT(cond, behavior) ((void)0)
#endif

#include "shmem.h"

#ifdef ENABLE_TIMER
#include "AscendTimer_device.hpp"
#endif

namespace Catccos::DGemm::Kernel::QuantA2AQp {

// Communication is submitted by CommBlock. Keep only the producer-ticket memory
// primitives used by this kernel for flag publication and polling.

__aicore__ inline uint64_t VolatileLoad64(__gm__ volatile uint64_t* addr) { return *addr; }

__aicore__ inline void DcciLine(__gm__ void* addr)
{
    AscendC::GlobalTensor<uint8_t> line;
    line.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(addr));
    __asm__ __volatile__("" ::: "memory");
    AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        line);
    __asm__ __volatile__("" ::: "memory");
}

// Each independently written flag must occupy its own cache line.
__aicore__ inline void PublishFlag(__gm__ volatile uint64_t* addr, uint64_t value)
{
    *addr = value;
    DcciLine(reinterpret_cast<__gm__ void*>(const_cast<__gm__ uint64_t*>(addr)));
}

} // namespace Catccos::DGemm::Kernel::QuantA2AQp

namespace Catccos::DGemm::Kernel {

namespace detail {

/// 总轮数 R = ⌈m/H⌉:把每卡 M 行按"每轮 H 行"切分的次数。
CATLASS_HOST_DEVICE inline uint32_t QuantA2ARoundCount(uint32_t m, uint32_t h)
{
    return h == 0 ? 0 : static_cast<uint32_t>((static_cast<uint64_t>(m) + h - 1) / h);
}

/// 第 round 轮的实际行数:整轮为 H,行尾轮收缩为 m - round*H;round 越界返回 0。
CATLASS_HOST_DEVICE inline uint32_t QuantA2AActualRoundRows(uint32_t m, uint32_t h, uint32_t round)
{
    const uint64_t completedRows = static_cast<uint64_t>(round) * h;
    if (completedRows >= m) {
        return 0;
    }
    const uint64_t remaining = static_cast<uint64_t>(m) - completedRows;
    return remaining < h ? static_cast<uint32_t>(remaining) : h;
}

// 完整的本地 MatMul 结果先写入对称 TX 区:每一轮都有自己独立的目标窗口,
// 因此不需要任何"消费确认"(ACK)——数据不会被覆盖,窗口之间互不重叠。
struct QuantA2ASymLayout {
    static constexpr uint32_t FLAG_BYTES = 64;
    static constexpr uint32_t MAX_PEERS = 32;

    /// 把字节数向上对齐到 64B(FLAG_BYTES)边界。
    static CATLASS_HOST_DEVICE uint64_t AlignFlag(uint64_t value)
    {
        return (value + FLAG_BYTES - 1) / FLAG_BYTES * FLAG_BYTES;
    }

    /// TX 区容量:每轮每个 peer 一个 (H, q) 行块,共 rounds × P 块。
    static CATLASS_HOST_DEVICE uint64_t
    TxBytes(uint32_t rounds, uint32_t p, uint32_t h, uint32_t q, uint32_t elementBytes)
    {
        return static_cast<uint64_t>(rounds) * p * h * q * elementBytes;
    }

    /// 输出区容量:Y=[P*m, q] 的 bf16 字节数。
    static CATLASS_HOST_DEVICE uint64_t YBytes(uint32_t m, uint32_t p, uint32_t q, uint32_t elementBytes)
    {
        return static_cast<uint64_t>(m) * p * q * elementBytes;
    }

    // Ready 协议工作区:2P 个"通知槽"(每 peer 一收一发)+ 1 个状态行。
    // TileHierarchicalReady::GetWorkspaceSize() 额外含一个 cache line 的
    // 对齐余量——aclshmem_align() 返回的偏移可能落在任意 64B 边界上。
    static CATLASS_HOST_DEVICE uint64_t ReadyBytes(uint32_t p)
    {
        return static_cast<uint64_t>(2 * p + 1) * FLAG_BYTES + FLAG_BYTES - 1;
    }

    // 生产票区:每个 (轮, peer, AIC) 一个标志、独占 cache line——
    // AIV 解锁一个 peer 的票时,不必等其他 peer 同轮完成。
    static CATLASS_HOST_DEVICE uint64_t ProdPeerDoneBytes(uint32_t rounds, uint32_t p, uint32_t aicNum)
    {
        return static_cast<uint64_t>(rounds) * p * aicNum * FLAG_BYTES;
    }

    // 对称区总容量 = TX(各轮) + Y(输出) + Ready 工作区 + producer tickets;
    // main.cpp 按 4K 对齐申请。
    static CATLASS_HOST_DEVICE uint64_t
    TotalBytes(uint32_t rounds, uint32_t p, uint32_t m, uint32_t h, uint32_t q, uint32_t elementBytes, uint32_t aicNum)
    {
        return AlignFlag(TxBytes(rounds, p, h, q, elementBytes) + YBytes(m, p, q, elementBytes) + ReadyBytes(p)) +
               ProdPeerDoneBytes(rounds, p, aicNum);
    }

    /// TX 内偏移:第 round 轮中"给 peer 的行块"的字节基址
    /// ((round*P + peer) * H * q * b)。
    static CATLASS_HOST_DEVICE uint64_t
    TxPeerOff(uint32_t round, uint32_t p, uint32_t peer, uint32_t h, uint32_t q, uint32_t elementBytes)
    {
        return (static_cast<uint64_t>(round) * p + peer) * h * q * elementBytes;
    }
};

// host 预生成的生产顺序:跨节点 peer 最先(最慢的 RDMA 传输最早启动)、
// 同节点 UDMA peer 其次、本卡最后(本地直写无需搬运,可与远端传输重叠)。
struct QuantA2APeerSchedule {
    static CATLASS_HOST_DEVICE void BuildTargetOrder(
        uint32_t rank, uint32_t p, uint32_t ranksPerNode, uint32_t (&order)[QuantA2ASymLayout::MAX_PEERS])
    {
        // 两趟扫描:第一趟(crossNode=1)收集跨节点 peer,第二趟(=0)收集
        // 同节点 peer,最后追加本卡——本卡永远排最后。
        uint32_t count = 0;
        const uint32_t localGroup = rank / ranksPerNode;
        for (uint32_t crossNode = 1; crossNode >= 0; --crossNode) {
            for (uint32_t peer = 0; peer < p; ++peer) {
                if (peer == rank || ((peer / ranksPerNode != localGroup) != (crossNode != 0))) {
                    continue;
                }
                order[count++] = peer;
            }
            if (crossNode == 0) {
                break;
            }
        }
        order[count++] = rank;
    }
};

// ENABLE_TIMER 的条件编译全部封装在这个空实现包装器里,
// 使 AIC/AIV 主路径在两种构建下保持同一份代码形状。
// 阶段:AIV 包络、AIC 计算段(Begin/EndCompute)、Copy 段、空闲段(Begin/EndWait)。
class QuantA2ATimer {
public:
    CATLASS_DEVICE QuantA2ATimer()
    {
#ifdef ENABLE_TIMER
        if (GetTimerBuffer() != nullptr) {
            timerTrace_ = true;
            timer_.Init(GetTimerBuffer(), AscendC::GetSystemCycle());
        }
#endif
    }

    CATLASS_DEVICE ~QuantA2ATimer()
    {
#ifdef ENABLE_TIMER
        if (timerTrace_) {
            timer_.Tok<Overwrite>(AscendTimer::KERNEL_TIMING_IDX);
        }
#endif
    }

    /// 开始 AIV 通信总包络计时(覆盖整个通信侧执行)。
    CATLASS_DEVICE void BeginAivEnvelope()
    {
#ifdef ENABLE_TIMER
        if (timerTrace_) {
            timer_.Tik(AscendTimer::AIV);
        }
#endif
    }

    /// 结束 AIV 通信总包络计时。
    CATLASS_DEVICE void EndAivEnvelope()
    {
#ifdef ENABLE_TIMER
        if (timerTrace_) {
            timer_.Tok<Overwrite>(AscendTimer::AIV);
        }
#endif
    }

    /// 开始一段 AIC 计算计时(重入安全:已激活则忽略)。
    CATLASS_DEVICE void BeginCompute()
    {
#ifdef ENABLE_TIMER
        if (timerTrace_ && !computeActive_) {
            timer_.Tik(AscendTimer::AIC);
            computeActive_ = true;
        }
#endif
    }

    /// 结束当前 AIC 计算段。
    CATLASS_DEVICE void EndCompute()
    {
#ifdef ENABLE_TIMER
        if (timerTrace_ && computeActive_) {
            timer_.Tok<Overwrite>(AscendTimer::AIC);
            computeActive_ = false;
        }
#endif
    }

    /// 开始 UDMA/RDMA 提交段计时(按引擎分槽,互不覆盖)。
    CATLASS_DEVICE void BeginCopy(bool isUdma)
    {
#ifdef ENABLE_TIMER
        const uint32_t index = isUdma ? 0U : 1U;
        if (timerTrace_ && !copyActive_[index]) {
            timer_.Tik(isUdma ? AscendTimer::AIV_LOCAL_COPY : AscendTimer::AIV_AG);
            copyActive_[index] = true;
        }
#else
        (void)isUdma;
#endif
    }

    /// 结束当前 UDMA/RDMA 提交段。
    CATLASS_DEVICE void EndCopy(bool isUdma)
    {
#ifdef ENABLE_TIMER
        const uint32_t index = isUdma ? 0U : 1U;
        if (timerTrace_ && copyActive_[index]) {
            timer_.Tok<Overwrite>(isUdma ? AscendTimer::AIV_LOCAL_COPY : AscendTimer::AIV_AG);
            copyActive_[index] = false;
        }
#else
        (void)isUdma;
#endif
    }

    /// 开始 AIV 等待段计时(忙等生产票)。
    CATLASS_DEVICE void BeginWait()
    {
#ifdef ENABLE_TIMER
        if (timerTrace_ && !waitActive_) {
            timer_.Tik(AscendTimer::AIV_WAIT);
            waitActive_ = true;
        }
#endif
    }

    /// 结束当前 AIV 等待段。
    CATLASS_DEVICE void EndWait()
    {
#ifdef ENABLE_TIMER
        if (timerTrace_ && waitActive_) {
            timer_.Tok<Overwrite>(AscendTimer::AIV_WAIT);
            waitActive_ = false;
        }
#endif
    }

private:
#ifdef ENABLE_TIMER
    AscendTimerDevice timer_;
    bool timerTrace_ = false;
    bool computeActive_ = false;
    bool waitActive_ = false;
    bool copyActive_[2] = {false, false};
#endif
};

/// 量化档:从 BlockMmad 提取 A/B 的 MX scale 元素与布局类型。
template <class BlockMmad, bool IsQuantized>
struct QuantA2AScaleTraits {
    using ElementA = uint8_t;
    using LayoutA = Catlass::layout::RowMajor;
    using ElementB = uint8_t;
    using LayoutB = Catlass::layout::RowMajor;
};

template <class BlockMmad>
struct QuantA2AScaleTraits<BlockMmad, true> {
    using ElementA = typename BlockMmad::ElementMxScaleA;
    using LayoutA = typename BlockMmad::TileCopy::LayoutMxScaleA;
    using ElementB = typename BlockMmad::ElementMxScaleB;
    using LayoutB = typename BlockMmad::TileCopy::LayoutMxScaleB;
};

/// 模板主形态(BF16 档落在这里):空类型——不创建 scale 张量。
template <bool IsQuantized, class Element, class Layout>
struct QuantA2AScaleTensorTraits {
    using Type = Catlass::EmptyClass;
};

/// 量化档:scale 的 tla 张量类型。
template <class Element, class Layout>
struct QuantA2AScaleTensorTraits<true, Element, Layout> {
    using Type = decltype(tla::MakeTensor(
        *static_cast<AscendC::GlobalTensor<Element>*>(nullptr), *static_cast<Layout*>(nullptr),
        Catlass::Arch::PositionGM{}));
};

} // namespace detail

using Catlass::GemmCoord;
using Catlass::MatrixCoord;

/// QuantMatmulAllToAll:量化/原生 MatMul + 沿 N 均匀 AllToAll(PUT 推模式)。
///
/// ------------------------------------------------------------------------
/// 新手导读(建议先读这段再看代码):
///   目标:每卡算 Y=A×B,按列切成 P 份,第 j 份送给 rank j,
///         最终每卡得到 Y=[P*m, N/P],行块 i 来自 rank i。
///   本版与主算子(GET 拉模式)的根本差异:**peer 粒度就绪的 PUT 推模式**。
///   一轮的时序:
///     1) AIC 每轮按跨机、同机、本卡顺序处理 peer:
///        self 列块直写 Y,远端列块写本卡对称 TX[round];
///     2) 每个 peer 的所有分片同步落 GM 后,各 AIC 立即发布该 peer 的完成票;
///     3) 对应 AIV 只轮询自己 peer 的完成票,就绪后立刻用 CommBlock PUT
///        整窗到对端 Y;某一 peer 就绪不再等待其它 peer 或其它 rank;
///     4) 尾部 Ready.Notify/WaitAll:确认所有 peer 推完,才能释放对称区。
///   窗口互不重叠,无 ACK。AIC 处理下一个 peer 时,上一 peer 的 UDMA/RDMA
///   搬运已经可以与 Cube 计算重叠。
/// ------------------------------------------------------------------------
template <class BlockMmad_, class BlockMmadScheduler_, class BlockComm_, class CommScheduler_, bool IsQuantized_>
class QuantMatmulAllToAll {
public:
    // BlockComm_    : CommBlock 字节搬运块(PUT 的执行体,按 transport 特化)
    // CommScheduler_: HierarchicalCommSwizzle——一 AIV 绑一 peer 并决定 transport
    using BlockMmad = BlockMmad_;
    using BlockMmadScheduler = BlockMmadScheduler_;
    using BlockComm = BlockComm_;
    using CommScheduler = CommScheduler_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementY = typename BlockMmad::ElementC;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutY = typename BlockMmad::LayoutC;
    using CopyTransport = Catccos::detail::CopyTransport;
    using Ready = Catccos::Comm::Tile::TileHierarchicalReady;

    static constexpr bool IS_QUANTIZED = IsQuantized_;
    using ScaleTraits = detail::QuantA2AScaleTraits<BlockMmad, IS_QUANTIZED>;
    using ElementMxScaleA = typename ScaleTraits::ElementA;
    using LayoutMxScaleA = typename ScaleTraits::LayoutA;
    using ElementMxScaleB = typename ScaleTraits::ElementB;
    using LayoutMxScaleB = typename ScaleTraits::LayoutB;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t READY_EVENT_ID = 0; // Ready 协议的事件编号
    // AIV 的 Ready scratch 紧跟 BlockComm 的 UB stage 之后(UB 布局见 AIV 侧)
    static constexpr uint32_t READY_UB_OFFSET = BlockComm::UB_STAGES * BlockComm::UB_STAGE_BYTES;

    static_assert(std::is_same_v<typename BlockComm::ElementSrc, uint8_t>);
    static_assert(std::is_same_v<typename BlockComm::ElementDst, uint8_t>);
    static_assert(READY_UB_OFFSET + Ready::UB_BYTES <= ArchTag::UB_SIZE);
    static_assert(READY_EVENT_ID < 8);

    struct Arguments {
        GemmCoord problemShape; // M rows on this rank, full N, K
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t ranksPerNode;
        uint32_t aicNum;
        uint32_t commInterval;
        uint32_t laneNum;
        uint32_t epoch; // 跨调用代数:Ready 协议用它区分不同调用
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrAScale;
        GM_ADDR ptrBScale;
        GM_ADDR ptrY;
        GM_ADDR ptrSymmetric;
    };

    struct Params {
        GemmCoord problemShape;
        uint32_t rankIdx;
        uint32_t rankSize;
        uint32_t ranksPerNode;
        uint32_t aicNum;
        uint32_t h;
        uint32_t laneNum;
        uint32_t epoch; // 跨调用代数(与 Arguments.epoch 一致)
        uint32_t targetOrder[detail::QuantA2ASymLayout::MAX_PEERS];
        __gm__ ElementA* ptrA;
        LayoutA layoutA;
        __gm__ ElementB* ptrB;
        LayoutB layoutB;
        __gm__ ElementMxScaleA* ptrAScale;
        LayoutMxScaleA layoutAScale;
        __gm__ ElementMxScaleB* ptrBScale;
        LayoutMxScaleB layoutBScale;
        __gm__ ElementY* ptrY;
        LayoutY layoutY;
        __gm__ uint8_t* txBase;
        GM_ADDR readyWorkspace;
        __gm__ volatile uint64_t* prodPeerDone;
    };

    /// 无独立 workspace——全部缓冲都来自对称区(见 ToUnderlyingArguments)。
    static size_t GetWorkspaceSize(Arguments const&) { return 0; }

    /// host 侧烘焙 Params:构建 layout、预生成 targetOrder
    /// (跨节点→同节点→本卡),并把对称区切出
    /// [TX 各轮][Y 输出][Ready 工作区][生产票] 四个子指针。
    static Params ToUnderlyingArguments(Arguments const& args, uint8_t* = nullptr)
    {
        Params p{};
        p.problemShape = args.problemShape;
        p.rankIdx = args.rankIdx;
        p.rankSize = args.rankSize;
        p.ranksPerNode = args.ranksPerNode;
        p.aicNum = args.aicNum;
        p.h = args.commInterval * L1_TILE_M;
        p.laneNum = args.laneNum;
        p.epoch = args.epoch;
        detail::QuantA2APeerSchedule::BuildTargetOrder(args.rankIdx, args.rankSize, args.ranksPerNode, p.targetOrder);

        auto layoutTagA = Catlass::layout::RowMajor::MakeLayout<ElementA>(args.problemShape.m(), args.problemShape.k());
        auto layoutTagB = Catlass::layout::RowMajor::MakeLayout<ElementB>(args.problemShape.k(), args.problemShape.n());
        auto layoutTagY = Catlass::layout::RowMajor::MakeLayout<ElementY>(
            static_cast<uint64_t>(args.problemShape.m()) * args.rankSize, args.problemShape.n() / args.rankSize);
        p.layoutA = tla::MakeLayoutFromTag(layoutTagA);
        p.layoutB = tla::MakeLayoutFromTag(layoutTagB);
        p.layoutY = tla::MakeLayoutFromTag(layoutTagY);

        if constexpr (IS_QUANTIZED) {
            const uint32_t mxScaleK = CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(args.problemShape.k());
            p.layoutAScale =
                tla::MakeMxScaleLayout<ElementMxScaleA, decltype(layoutTagA), false>(args.problemShape.m(), mxScaleK);
            p.layoutBScale =
                tla::MakeMxScaleLayout<ElementMxScaleB, decltype(layoutTagB), true>(mxScaleK, args.problemShape.n());
        }

        p.ptrA = reinterpret_cast<__gm__ ElementA*>(args.ptrA);
        p.ptrB = reinterpret_cast<__gm__ ElementB*>(args.ptrB);
        p.ptrAScale = reinterpret_cast<__gm__ ElementMxScaleA*>(args.ptrAScale);
        p.ptrBScale = reinterpret_cast<__gm__ ElementMxScaleB*>(args.ptrBScale);
        p.ptrY = reinterpret_cast<__gm__ ElementY*>(args.ptrY);

        const uint32_t q = args.problemShape.n() / args.rankSize;
        const uint32_t rounds = detail::QuantA2ARoundCount(args.problemShape.m(), p.h);
        auto* sym = reinterpret_cast<__gm__ uint8_t*>(args.ptrSymmetric);
        p.txBase = sym;
        __gm__ uint8_t* cur = sym + detail::QuantA2ASymLayout::TxBytes(rounds, args.rankSize, p.h, q, sizeof(ElementY));
        p.ptrY = reinterpret_cast<__gm__ ElementY*>(cur);
        cur += detail::QuantA2ASymLayout::YBytes(args.problemShape.m(), args.rankSize, q, sizeof(ElementY));
        p.readyWorkspace = cur;
        cur += detail::QuantA2ASymLayout::ReadyBytes(args.rankSize);
        cur = reinterpret_cast<__gm__ uint8_t*>(
            (reinterpret_cast<uintptr_t>(cur) + detail::QuantA2ASymLayout::FLAG_BYTES - 1) &
            ~(static_cast<uintptr_t>(detail::QuantA2ASymLayout::FLAG_BYTES) - 1));
        p.prodPeerDone = reinterpret_cast<__gm__ volatile uint64_t*>(cur);
        return p;
    }

    CATLASS_DEVICE QuantMatmulAllToAll() = default;

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    /// AIC(计算侧):逐轮按 targetOrder 处理 peer。跨机 peer 先算、同机 peer 次之、
    /// 本卡最后；某个 peer 的全部 AIC 分片完成后立即发布该 peer 的完成票。
    /// 后续 peer 的计算可与前一个 peer 的 UDMA/RDMA 搬运重叠。
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        const uint32_t aicoreIdx = AscendC::GetBlockIdx();
        const uint32_t aicoreNum = params.aicNum;
        const GemmCoord problem = params.problemShape;
        const GemmCoord mmadBlockShape{L1_TILE_M, L1_TILE_N, L1_TILE_K};
        const uint32_t q = problem.n() / params.rankSize;
        const uint32_t rounds = detail::QuantA2ARoundCount(problem.m(), params.h);

        BlockMmad blockMmad(resource);
        AscendC::GlobalTensor<ElementA> gmA;
        AscendC::GlobalTensor<ElementB> gmB;
        AscendC::GlobalTensor<ElementY> gmY;
        gmA.SetGlobalBuffer(params.ptrA);
        gmB.SetGlobalBuffer(params.ptrB);
        gmY.SetGlobalBuffer(params.ptrY);

        auto tensorA = tla::MakeTensor(gmA, params.layoutA, Catlass::Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(gmB, params.layoutB, Catlass::Arch::PositionGM{});
        auto tensorY = tla::MakeTensor(gmY, params.layoutY, Catlass::Arch::PositionGM{});

        using ScaleTensorA =
            typename detail::QuantA2AScaleTensorTraits<IS_QUANTIZED, ElementMxScaleA, LayoutMxScaleA>::Type;
        using ScaleTensorB =
            typename detail::QuantA2AScaleTensorTraits<IS_QUANTIZED, ElementMxScaleB, LayoutMxScaleB>::Type;
        ScaleTensorA tensorScaleA{};
        ScaleTensorB tensorScaleB{};
        if constexpr (IS_QUANTIZED) {
            AscendC::GlobalTensor<ElementMxScaleA> gmScaleA;
            AscendC::GlobalTensor<ElementMxScaleB> gmScaleB;
            gmScaleA.SetGlobalBuffer(params.ptrAScale);
            gmScaleB.SetGlobalBuffer(params.ptrBScale);
            tensorScaleA = tla::MakeTensor(gmScaleA, params.layoutAScale, Catlass::Arch::PositionGM{});
            tensorScaleB = tla::MakeTensor(gmScaleB, params.layoutBScale, Catlass::Arch::PositionGM{});
        }

        // TX 按 (H, q) 行主序视图;每轮每 peer 的行块用 TxPeerOff 再偏移。
        const auto layoutTagRound = Catlass::layout::RowMajor::MakeLayout<ElementY>(params.h, q);
        const auto layoutRound = tla::MakeLayoutFromTag(layoutTagRound);
        AscendC::GlobalTensor<ElementY> gmTX;
        gmTX.SetGlobalBuffer(reinterpret_cast<__gm__ ElementY*>(params.txBase));

        // ---- 逐轮计算:第 round 轮负责行 [round*H, round*H+actualH) ----
        // AIC 全程零等待:不等 AIV、不等对端;每完成一个 peer 就发完成票,
        // 对应 AIV 即可开始推该窗,与后续 peer 的计算重叠。
        for (uint32_t round = 0; round < rounds; ++round) {
            const uint32_t actualH = detail::QuantA2AActualRoundRows(problem.m(), params.h, round);
            const GemmCoord actualProblemShape{actualH, problem.n(), problem.k()};
            BlockMmadScheduler mmadScheduler(actualProblemShape, mmadBlockShape.GetCoordMN());
            const uint32_t coreLoops = mmadScheduler.GetCoreLoops();
            const MatrixCoord roundOffset{round * params.h, 0};

            timer.BeginCompute();
            // ---- 按 targetOrder 逐 peer 生产:跨节点最先、同节点次之、本卡最后 ----
            for (uint32_t orderIdx = 0; orderIdx < params.rankSize; ++orderIdx) {
                const uint32_t peer = params.targetOrder[orderIdx];
                bool produced = false; // 本核是否分到该 peer 的 tile(决定要不要排空流水)
                for (uint32_t blockIdx = aicoreIdx; blockIdx < coreLoops; blockIdx += aicoreNum) {
                    const GemmCoord blockCoord = mmadScheduler.GetBlockCoord(blockIdx);
                    const GemmCoord actualBlockShape = mmadScheduler.GetActualBlockShape(blockCoord);
                    const GemmCoord offsetCoord = blockCoord * mmadBlockShape;
                    const MatrixCoord offsetMN = offsetCoord.GetCoordMN();
                    // 本核是否负责该 peer 的 tile
                    if (offsetMN.column() / q != peer) {
                        continue;
                    }
                    produced = true;
                    const auto blockOffsetA = offsetCoord.GetCoordMK() + roundOffset;
                    const auto blockOffsetB = offsetCoord.GetCoordKN();

                    auto tensorBlockA = GetTile(
                        tensorA, tla::MakeCoord(blockOffsetA[0], blockOffsetA[1]),
                        tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                    auto tensorBlockB = GetTile(
                        tensorB, tla::MakeCoord(blockOffsetB[0], blockOffsetB[1]),
                        tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));

                    if (peer == params.rankIdx) {
                        // self:直写 Y 的 self 行块(行 rankIdx*m + round*H 起),不出卡。
                        const auto outputOffset = tla::MakeCoord(
                            static_cast<uint64_t>(params.rankIdx) * problem.m() + round * params.h + offsetMN.row(),
                            offsetMN.column() - peer * q);
                        auto tensorBlockY =
                            GetTile(tensorY, outputOffset, tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                        if constexpr (IS_QUANTIZED) {
                            auto tensorBlockScaleA = GetTile(
                                tensorScaleA,
                                tla::MakeCoord(blockOffsetA[0], blockOffsetA[1] / Catlass::MX_SCALE_GROUP_NUM),
                                tla::MakeShape(
                                    actualBlockShape.m(), CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(actualBlockShape.k())));
                            auto tensorBlockScaleB = GetTile(
                                tensorScaleB,
                                tla::MakeCoord(blockOffsetB[0] / Catlass::MX_SCALE_GROUP_NUM, blockOffsetB[1]),
                                tla::MakeShape(
                                    CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(actualBlockShape.k()), actualBlockShape.n()));
                            blockMmad(
                                tensorBlockA, tensorBlockB, tensorBlockY, actualBlockShape, tensorBlockScaleA,
                                tensorBlockScaleB);
                        } else {
                            blockMmad(tensorBlockA, tensorBlockB, tensorBlockY, actualBlockShape);
                        }
                    } else {
                        // remote:写 TX[round] 的 peer 行块,供本卡对应 AIV 立即 PUT。
                        const uint64_t txElementOffset =
                            detail::QuantA2ASymLayout::TxPeerOff(
                                round, params.rankSize, peer, params.h, q, sizeof(ElementY)) /
                            sizeof(ElementY);
                        const auto txOffset = tla::MakeCoord(offsetMN.row(), offsetMN.column() - peer * q);
                        auto gmTXBlock = gmTX[txElementOffset];
                        auto tensorTXBlock = tla::MakeTensor(gmTXBlock, layoutRound, Catlass::Arch::PositionGM{});
                        auto tensorBlockTX = GetTile(
                            tensorTXBlock, txOffset, tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                        if constexpr (IS_QUANTIZED) {
                            auto tensorBlockScaleA = GetTile(
                                tensorScaleA,
                                tla::MakeCoord(blockOffsetA[0], blockOffsetA[1] / Catlass::MX_SCALE_GROUP_NUM),
                                tla::MakeShape(
                                    actualBlockShape.m(), CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(actualBlockShape.k())));
                            auto tensorBlockScaleB = GetTile(
                                tensorScaleB,
                                tla::MakeCoord(blockOffsetB[0] / Catlass::MX_SCALE_GROUP_NUM, blockOffsetB[1]),
                                tla::MakeShape(
                                    CeilDiv<Catlass::MX_SCALE_GROUP_NUM>(actualBlockShape.k()), actualBlockShape.n()));
                            blockMmad(
                                tensorBlockA, tensorBlockB, tensorBlockTX, actualBlockShape, tensorBlockScaleA,
                                tensorBlockScaleB);
                        } else {
                            blockMmad(tensorBlockA, tensorBlockB, tensorBlockTX, actualBlockShape);
                        }
                    }
                }

                // 等本 AIC 为该 peer 提交的 MMAD/写回完成,再发布独立完成票。
                // 未分到 tile 的 AIC 也立即发布,使 AIV 无需等待空闲 AIC。
                if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                    if (produced) {
                        blockMmad.SynchronizeBlock();
                    }
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                PublishAicPeerFlag(params, round, peer, aicoreIdx, params.epoch);
            }
            timer.EndCompute();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    /// AIV(通信侧):一 AIV 绑一 peer(CommScheduler 绑定)。每轮只轮询该 peer
    /// 的全部 AIC 完成票,不等待其它 peer、其它 rank,也不执行全 rank barrier;
    /// 到齐后立即把 TX[round] 中的整窗 PUT 推到对端 Y。
    /// 尾部 Ready.Notify/WaitAll 收账:所有 peer 推完才可释放对称区。
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        const uint32_t aivIdx = AscendC::GetBlockIdx();
        const uint32_t q = params.problemShape.n() / params.rankSize;
        const uint32_t rowBytes = q * sizeof(ElementY);
        const uint32_t rounds = detail::QuantA2ARoundCount(params.problemShape.m(), params.h);
        // 绑定 peer:aivIdx 即 peer 序号(一 AIV 一 peer);transport 由节点归属决定。
        CommScheduler commScheduler({params.h, rowBytes, params.rankSize}, params.rankIdx, aivIdx);

        typename BlockComm::Params blockParams{};
        if constexpr (BlockComm::IsDynamic) {
            blockParams.blockShape = commScheduler.GetActualBlockShape({0U, 0U});
        }
        BlockComm blockCopy(resource, blockParams);
        Ready ready(params.readyWorkspace, params.rankIdx, params.rankSize);
        auto readyScratch = reinterpret_cast<__ubuf__ uint32_t*>(
            resource.ubBuf.template GetBufferByByte<uint8_t>(READY_UB_OFFSET).GetPhyAddr());

        timer.BeginAivEnvelope();
        for (uint32_t round = 0; round < rounds; ++round) {
            if (!commScheduler.IsActiveCore()) {
                continue;
            }
            const uint32_t peer = commScheduler.GetPeer();
            // ① 忙等本轮该 peer 的全部 AIC 生产票(刻意自旋:就绪即刻推窗)
            timer.BeginWait();
            PollPeerProduction(params, round, peer, params.epoch);
            timer.EndWait();
            // ② self 轮:AIC 已把本卡列块直写 Y,无搬运
            if (peer == params.rankIdx) {
                continue;
            }
            // ③ 推送整窗:本地生产票已齐,PUT 直接起飞(无跨 rank 握手);
            //    多轮的 PUT 在 QP 中异步在途,由尾部 Notify 的 quiet 统一确认。
            const uint32_t actualH = detail::QuantA2AActualRoundRows(params.problemShape.m(), params.h, round);
            const CopyTransport transport = commScheduler.GetTransport();
            timer.BeginCopy(transport == CopyTransport::Udma);
            CopyWindow(blockCopy, params, peer, transport, round, actualH);
            timer.EndCopy(transport == CopyTransport::Udma);
        }

        // ---- 尾部收账 ----
        // 活跃 AIV 通知对端"我的全部轮次已推完"(Ready 协议,带 epoch 防跨调用混淆):
        // 先 quiet 该 peer 的全部 payload QP(在途 PUT 全部送达),再经 QP0 推通知;
        // 仅 AIV0 等待所有对端的通知——之后本卡对称 TX 区才可安全释放/复用。
        if (commScheduler.IsActiveCore()) {
            ready.Notify(
                commScheduler.GetPeer(), commScheduler.GetTransport(), params.epoch, params.laneNum, readyScratch,
                READY_EVENT_ID);
        }
        if (aivIdx == 0) {
            ready.WaitAll(params.epoch);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        timer.EndAivEnvelope();
    }

private:
    static CATLASS_DEVICE __gm__ volatile uint64_t* ProdPeerDoneSlot(
        Params const& params, uint32_t round, uint32_t peer, uint32_t aic)
    {
        /// 生产票寻址:[round][peer][aic] 三级,每票独占 64B cache line。
        constexpr uint64_t wordsPerFlag = detail::QuantA2ASymLayout::FLAG_BYTES / sizeof(uint64_t);
        const uint64_t index = ((static_cast<uint64_t>(round) * params.rankSize + peer) * params.aicNum) + aic;
        return params.prodPeerDone + index * wordsPerFlag;
    }

    /// 发布"本 AIC 对 (round, peer) 的份额已完成":写 ticket 并刷新 cache line。
    static CATLASS_DEVICE void PublishAicPeerFlag(
        Params const& params, uint32_t round, uint32_t peer, uint32_t aic, uint64_t ticket)
    {
        QuantA2AQp::PublishFlag(ProdPeerDoneSlot(params, round, peer, aic), ticket);
    }

    /// 忙等"round 轮 peer 的全部 AIC 生产票"到位:每票先 DcciLine 刷新
    /// 再 volatile 读,>= ticket 即该 AIC 的份额完成。刻意不睡眠——
    /// AIV 就绪后立刻推窗,不让已就绪的数据干等。
    static CATLASS_DEVICE void PollPeerProduction(Params const& params, uint32_t round, uint32_t peer, uint64_t ticket)
    {
        for (uint32_t aic = 0; aic < params.aicNum; ++aic) {
            auto* flag = ProdPeerDoneSlot(params, round, peer, aic);
            do {
                QuantA2AQp::DcciLine(reinterpret_cast<__gm__ void*>(const_cast<__gm__ uint64_t*>(flag)));
            } while (QuantA2AQp::VolatileLoad64(flag) < ticket);
        }
    }

    /// 推送一个通信窗:src = 本卡 TX[round] 里给 peer 的整窗(actualH×q 行),
    /// dst = 对端 Y 的 peer 行块第 round 轮窗口;整窗交给 BlockComm,
    /// 由它按 transport(UDMA/RDMA)与 laneNum 拆分到各数据 QP 执行。
    CATLASS_DEVICE void CopyWindow(
        BlockComm& blockCopy, Params const& params, uint32_t peer, CopyTransport transport, uint32_t round,
        uint32_t actualH) const
    {
        const uint32_t q = params.problemShape.n() / params.rankSize;
        const uint32_t rowBytes = q * sizeof(ElementY);
        // src:本卡 TX[round] 的 peer 行块(TxPeerOff 定位);
        // dst:对端 Y 的 peer 行块第 round 轮窗口(行 rankIdx*m + round*H 起)。
        const uint64_t srcOffset =
            detail::QuantA2ASymLayout::TxPeerOff(round, params.rankSize, peer, params.h, q, sizeof(ElementY));
        const uint64_t dstOffset =
            static_cast<uint64_t>(static_cast<uint64_t>(params.rankIdx) * params.problemShape.m() + round * params.h) *
            rowBytes;

        const MatrixCoord windowShape{actualH, rowBytes};
        const Catlass::layout::RowMajor windowLayout{actualH, rowBytes};
        AscendC::GlobalTensor<uint8_t> gmSrc;
        AscendC::GlobalTensor<uint8_t> gmDst;
        gmSrc.SetGlobalBuffer(params.txBase + srcOffset);
        gmDst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(params.ptrY) + dstOffset);
        blockCopy(gmSrc, windowLayout, gmDst, windowLayout, windowShape, peer, transport, params.laneNum);
    }

    Catlass::Arch::Resource<ArchTag> resource;
    detail::QuantA2ATimer timer;
};

} // namespace Catccos::DGemm::Kernel

#endif // CATCCOS_DGEMM_KERNEL_ASCEND950_QUANT_MATMUL_ALLTOALL_HPP
