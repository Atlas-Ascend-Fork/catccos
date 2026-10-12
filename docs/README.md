# CATCCOS 文档

CATCCOS（**CA**NN **T**emplates for **C**ompute-**C**ommunication **O**verlap **S**ubroutines）是昇腾计算-通信融合算子模板库，基于 CATLASS 分层架构扩展多卡通信能力，提供计算与集合通信细粒度并行的融合算子模板及可运行样例。

## 快速导航

| 文档 | 说明 |
|------|------|
| [快速上手](quickstart.md) | 从零开发融合算子的完整 walkthrough |
| [API 参考](api/api.md) | Kernel / Block / Tile / Basic 分层 API，Host/Device 接口 |
| [算子总览](operators.md) | 当前支持的全部融合算子清单 |
| [实现模式](operators/implementation_patterns.md) | 按通信/计算模式分类的通用实现说明 |
| [CommSwizzle 算法](comm_swizzle/comm_swizzle_algorithm.md) | 通信调度与计算-通信 overlap 策略 |
| [Pre-commit 指南](pre-commit-guide.md) | 代码检查与提交规范 |

## 模板分层架构

CATCCOS 沿 CATLASS 风格分为四层，自顶向下组合：

| 层级 | 命名空间 / 路径 | 职责 |
|------|----------------|------|
| **Kernel** | `Catccos::DGemm::Kernel::*`、`Catccos::Comm::Kernel::*` | 完整融合算子，编排计算与通信流水线 |
| **Block** | `Catccos::DGemm::Block::*`、`Catccos::Comm::Block::*` | 单核 MMAD 或通信块逻辑及调度 |
| **Tile** | `Catccos::Comm::Tile::*` | 远端数据搬运（Put/Get，MTE/RDMA） |
| **Basic** | `AscendC::*`、ACLSHMEM | 硬件指令与对称内存原语 |

头文件目录：

```text
include/catccos/
├── catccos.hpp           # 核心工具（CommSwizzle、坐标辅助）
├── dgemm/                # 分布式 GEMM 融合
│   ├── kernel/           # 融合 Kernel 模板
│   ├── block/            # 计算调度与 swizzle
│   └── device/           # DeviceDGemm 主机侧启动器
├── comm/                 # 远端内存与集合通信
│   ├── block/
│   ├── tile/
│   └── kernel/           # 纯通信 Kernel
├── epilogue/             # 量化/反量化 epilogue
└── detail/               # CopyDirect、CopyTransport 等枚举
```

分层示意图见项目根目录 [README](../README.md) 中的 `docs/images/api_level.png`。

## 硬件平台

| 平台标识 | 说明 |
|---------|------|
| **Atlas A2/A3** | 标注为 `Atlas A2/A3` 的算子同时支持 Atlas A2 与 Atlas A3 |
| **Atlas 350** | 标注为 `Atlas 350` 的算子仅支持 Atlas 350 平台，示例目录前缀为 `ascend950_*` |

算子与平台的对应关系见 [算子总览](operators.md)。

## 开发路径

当前文档以 **C++ 直调（examples）** 为主：每个算子样例提供 `*_host.h`、`*_device.h`、`main.cpp` 及 `scripts/build.sh` / `run.sh`，可直接编译运行验证。

典型流程：

1. 阅读 [算子总览](operators.md)，选择目标算子类别
2. 参考 [examples/CMakeLists.txt](../examples/CMakeLists.txt) 中注册的样例代码
3. 查阅 [API 参考](api/api.md) 与 [实现模式](operators/implementation_patterns.md) 理解分层组装方式
4. 按 [快速上手](quickstart.md) 开发新算子或定制现有模板

## 相关目录

| 目录 | 说明 |
|------|------|
| [examples/](../examples/CMakeLists.txt) | 48 个 C++ 算子样例入口（含条件构建的 RDMA、独立构建的 HCCL 变体）和 1 个条件构建的 PyTorch binding，详见[算子总览](operators.md) |
| [include/catccos/](../include/catccos/catccos.hpp) | 模板头文件（header-only 库） |
| [utils/](../utils/info.h) | 共享 Host 代码（算子注册、SHMEM 初始化等） |
| [tools/](../tools/README.md) | AscendTimer 性能采集工具 |
