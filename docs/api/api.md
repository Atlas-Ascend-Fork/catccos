# CATCCOS DGemm组件

CATCCOS的分层结构如下所示(以AllGather Matmul为例)，这些组件针对数据类型、数据排布和数学指令进行特化。

| API 层级             | API 类 和/或 函数 名称                   |
| ---                  | ---                                               |
| Kernel               | `Catccos::DGemm::Kernel::AllGatherMatmul`            |
| Block           | `Catlass::Gemm::Block::BlockMmad` <br /> `Catccos::Comm::Block::CommBlock` <br /> `Catccos::DGemm::Block::GemmBlockSwizzleAllGatherMesh` <br /> `Catccos::Comm::Block::BlockCommSwizzle` <br /> |
| Tile | `TileRemoteCopy` <br /> |
| Basic                 | `AscendC::DataCopy` |

在CATCCOS中，我们为每个算子定义Host和Device两个模块。Host模块主要负责**算子执行的输入输出处理**，包括Device地址空间申请、输入数据构造、输出结果保存等功能。Device模块则负责汇总算子各个入参，然后**调用Kernel层级的API执行算子**。

具体可见CATCCOS的示例中[examples/allgather_matmul](../../examples/allgather_matmul/README.md)，如下文摘录所示。

```c++
// examples/allgather_matmul/allgather_matmul_host.h
class AllGatherMatmulOperator : public CatccosOperator {
public:
    // 按 cocTiling 分配 Device 内存；dataFile 非空时读取输入，否则使用全 1 输入。
    void AllocateDeviceSpace(
        KernelParams& params, const CocTilingParams& cocTiling, uint32_t rankId, std::string dataFile) override;

    // 将 Device 结果拷贝到 Host 并保存，供结果验证使用。
    void WriteResultFile(
        const KernelParams& params, const CocTilingParams& cocTiling, uint32_t rankId, std::string dataFile) override;
};
```

Device API调用说明

示例在 `examples/allgather_matmul/allgather_matmul_device.h` 的 `AllGatherMatmulConfig` 中组装计算和通信组件。下面的片段为结构示意，完整的类型定义与 padding 配置请参考该头文件。

```c++
// 第一步: 创建block层mmad
// 参数
using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<true>;
using L1TileShape = Catlass::GemmShape<M0, N0, K0>;
using L0TileShape = Catlass::GemmShape<M0, N0, 64>;

using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;
using PaddingHelperB = typename Catccos::Padding::PaddingHelper<BType, EnablePadding_>;
using ActualTypeB = typename PaddingHelperB::ActualType;
using GlobalPaddingB = typename PaddingHelperB::GlobalPadding;
using BlockMmad = Catlass::Gemm::Block::BlockMmad<
    MmadDispatchPolicy, L1TileShape, L0TileShape, AType, ActualTypeB, CType
>;

// 第二步：指定计算时的数据走位方式
using BlockMmadScheduler = Catccos::DGemm::Block::GemmBlockSwizzleAllGatherMesh<7, 1>;

// 第三步: 创建block层AllGather
// 参数
using RemoteSrcType = AType;
using RemoteDstType = AType;
using CopyDirect = Catccos::detail::CopyDirect;
// 指定数据搬运的方向和方式
using CopyTransport = Catccos::detail::CopyTransport;
using TileRemoteCopy = Comm::Tile::TileRemoteCopy<ArchTag, IS_DYNAMIC, RemoteSrcType, RemoteDstType, void, CopyDirect::Put, CopyTransport::Mte>;
using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

using CommDispatchPolicy = Comm::AtlasCommRemoteCopy<ArchTag, UB_STAGES, IS_DYNAMIC>;
using BlockComm = Comm::Block::CommBlock<
    CommDispatchPolicy,
    RemoteSrcType, RemoteDstType,
    void,
    TileRemoteCopy, TileScheduler
>;

// 第四步：指定通信时的数据搬运策略
using BlockCommScheduler = Catccos::Comm::Block::BlockCommSwizzle<IS_DYNAMIC, void, 0>;

// 第五步：在kernel层将mmad和AllGather组合到一起
using AllGatherMatmulKernel = DGemm::Kernel::AllGatherMatmul<
    GlobalPaddingB,
    BlockMmad,
    BlockComm,
    BlockMmadScheduler,
    BlockCommScheduler,
    WORKSPACE_STAGES
>;

// 第六步：通过 DeviceDGemm 启动配置好的 Kernel。
using DeviceOp = Catccos::DGemm::Device::DeviceDGemm<AllGatherMatmulKernel>;
```

`DeviceOp::Arguments` 的构造、`Initialize` 和 `Run` 调用可参考 [allgather_matmul_wrapper.cpp](../../examples/allgather_matmul/allgather_matmul_wrapper.cpp) 的 `RunAgmmDeviceOp`。


## Kernel API

Kernel层API是设备侧调用的入口，也是融合连续矩阵乘、通信数据搬运或其他操作的组合点。代码路径在 `include/catccos/dgemm/kernel` 下。下面展示 `allgather_matmul.hpp` 中的声明；上面的示例使用 `allgather_matmul_with_local_mm_opt.hpp` 中带 `GlobalPaddingB` 参数的变体。

```cpp
namespace Catccos::DGemm::Kernel {
template <
    class BlockMmad_,
    class BlockComm_,
    class BlockMmadScheduler_,
    class BlockCommScheduler_,
    uint32_t WORKSPACE_STAGES_
>
class AllGatherMatmul;
} // namespace Catccos::DGemm::Kernel
```

## Block API

Block API包括“矩阵乘累加”和“通信算子数据搬运”两部分，其中矩阵计算使用catlass提供的BlockMmad接口，block层指定的矩阵分块计算策略在include/catccos/dgemm/block下定义。通信算子的数据搬运接口和搬运策略代码实现在include/catccos/comm/block下，支持远端读写及本地读写操作。

### 通信数据搬运

```c++
// include/catccos/comm/block/comm_block_remote_copy.hpp
template <
    class ArchTag_,
    uint32_t UB_STAGES_,
    bool IsDynamic_,
    class SrcType_,
    class DstType_,
    class BlockShape_,
    class TileRemoteCopy_,
    class TileSwizzle_
>
class CommBlock <
    AtlasCommRemoteCopy<ArchTag_, UB_STAGES_, IsDynamic_>,
    SrcType_,
    DstType_,
    BlockShape_,
    TileRemoteCopy_,
    TileSwizzle_
>;
```

## Tile API

Tile粒度的数据搬运实现在include/catccos/comm/tile下，支持put和get两种数据搬运方向(CopyDirect设置)、mte和rdma两种数据搬运模式(CopyTransport设置)。

```cpp
template <
    class ArchTag,
    bool IsDynamic_, // tileshape参数是否为动态的
    class SrcType_,
    class DstType_,
    class TileShape_,
    detail::CopyDirect CopyDirect_,
    detail::CopyTransport CopyTransport_
>
struct TileRemoteCopy;
```

## Basic API

Basic层级API封装了实际的硬件指令调用，这些指令加速了MMAD或数据拷贝操作，实现了对硬件能力的抽象，开放了芯片能力，保证了完备性和兼容性，其中ISASI类API，不保证跨硬件版本兼容。

## Host API

Host 侧负责算子 I/O 与参数校验，通过 `CatccosOperator` 基类和算子注册机制与 Device 侧解耦。

### CatccosOperator

定义于 `utils/catccos_operator.h`，各算子在 `*_host.h` 中继承并实现以下虚函数：

| 虚函数 | 说明 |
|--------|------|
| `CheckCocTilingParams` | 校验 tiling 参数合法性 |
| `AllocateDeviceSpace` | 申请 Device 内存并加载输入数据 |
| `WriteResultFile` | 将计算结果 D2H 并保存 |
| `GetWorkspaceSize` | 返回 workspace / padding 字节数 |
| `GetActualKernelType` | 返回 `CocCommType` 枚举值 |

### 算子注册

```cpp
// utils/operator_registry.h
REGISTER_OPERATOR("AllGatherMatmul", AllGatherMatmulOperator);

// main.cpp 中使用
auto op = OperatorRegistry::Instance().CreateOperator("AllGatherMatmul");
```

`REGISTER_OPERATOR` 宏在静态初始化阶段将算子名与工厂函数注册到 `OperatorRegistry` 单例。

### 公共参数结构

- `CocTilingParams`（`utils/info.h`）：问题规模（m/k/n）、tile 大小（m0/k0/n0）、通信 tiling 参数
- `KernelParams`（`utils/info.h`）：Device 侧指针（ptrA/ptrB/ptrC）及 custom 指针数组

## Device API

Device 侧通过 `DeviceDGemm` 模板类启动 Kernel，定义于 `include/catccos/dgemm/device/device_dgemm.hpp`。

```cpp
namespace Catccos::DGemm::Device {
template <class DGemmKernel>
class DeviceDGemm {
public:
    using Arguments = typename DGemmKernel::Arguments;
    using Params = typename DGemmKernel::Params;

    static size_t GetWorkspaceSize(Arguments const &args);
    Catlass::Status Initialize(Arguments const &args, uint8_t *workspace = nullptr);
    Catlass::Status Run(aclrtStream stream, uint32_t blockDim, uint64_t fftsAddr);
};
}
```

典型调用流程：`GetWorkspaceSize` → `Initialize` → `Run`。Kernel 入口由 `KernelAdapter`（`kernel_adapter.hpp`）通过 `<<<>>>` 直调。

## Comm API 索引

通信相关枚举定义于 `include/catccos/detail/remote_copy_type.hpp`：

| 枚举 | 取值 | 说明 |
|------|------|------|
| `CopyDirect` | `Put`, `Get` | 数据搬运方向：主动写入远端 / 从远端读取 |
| `CopyTransport` | `Mte`, `Rdma` | 搬运通道：MTE 本地引擎 / RDMA 远端直写 |

`TileRemoteCopy` 通过模板参数指定 `CopyDirect` 和 `CopyTransport`；`CommBlock` 封装 block 级通信逻辑，详见上文 Tile API 与 Block API 章节。

通信 block 实现位于 `include/catccos/comm/block/`，tile 实现位于 `include/catccos/comm/tile/`。

## Kernel 目录索引

完整算子语义与分类见 [operators.md](../operators.md)。以下为 Kernel 头文件清单。

### DGemm Kernel（`include/catccos/dgemm/kernel/`）

| 头文件 | 融合模式 |
|--------|---------|
| `allgather_matmul.hpp` | AllGather + MatMul |
| `allgather_matmul_with_local_mm_opt.hpp` | AllGather + MatMul（本地 MM 优化） |
| `allgather_matmul_with_gather_result.hpp` | AllGather + MatMul（输出 Gather 结果） |
| `allgather_matmul_with_gather_result_and_local_mm.hpp` | 上述 + 本地 MM 优化 |
| `allgather_matmul_with_remote_read.hpp` | AllGather + MatMul（Remote Read） |
| `allgather_matmul_with_remote_read_local_mm_opt.hpp` | Remote Read + 本地 MM 优化 |
| `allgather_matmul_with_rdma_write.hpp` | AllGather + MatMul（RDMA Write） |
| `allgather_matmul_dequant.hpp` | AllGather + MatMul + Dequant |
| `allgather_matmul_dequant_bias.hpp` | AllGather + MatMul + Dequant + Bias |
| `matmul_allreduce.hpp` | MatMul + AllReduce |
| `matmul_reduce_scatter.hpp` | MatMul + ReduceScatter |
| `matmul_reduce_scatter_tla.hpp` | MatMul + ReduceScatter（TLA） |
| `matmul_reduce_scatter_mx_tla.hpp` | MX-FP8 MatMul + ReduceScatter |
| `matmul_dequant_reduce_scatter_v2.hpp` | Dequant + MatMul + ReduceScatter |
| `grouped_matmul_alltoallv.hpp` | GroupedMatMul + AllToAllV |
| `grouped_matmul_alltoallv_tla.hpp` | GroupedMatMul + AllToAllV（TLA） |
| `grouped_matmul_alltoallv_mx.hpp` | MX GroupedMatMul + AllToAllV |
| `alltoallv_grouped_matmul.hpp` | AllToAllV + GroupedMatMul |
| `alltoallv_gmm_v2.hpp` | AllToAllV + GMM v2 |
| `alltoallv_gmm_dequant_v2.hpp` | AllToAllV + Dequant GMM v2 |
| `gmm_alltoallv_v2.hpp` | GMM + AllToAllV v2 |
| `ascend950_allgather_matmul.hpp` | Atlas 350 AllGather + MatMul |
| `ascend950_alltoall_matmul.hpp` | Atlas 350 AllToAll + MatMul |
| `ascend950_matmul_alltoall.hpp` | Atlas 350 MatMul + AllToAll |
| `mx_allgather_matmul.hpp` | MX AllGather + MatMul |

### Comm Kernel（`include/catccos/comm/kernel/`）

| 头文件 | 融合模式 |
|--------|---------|
| `quant_allgather.hpp` | Quant AllGather（BF16 → HiF8） |
| `quant_alltoall.hpp` | Quant AllToAll（BF16 → HiF8） |
| `mx_quant_allgather.hpp` | MX Quant + AllGather |
