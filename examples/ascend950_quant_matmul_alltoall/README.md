# Ascend950 QuantMatmul AllToAll

## 功能说明

- 算子功能：完成量化 MatMul 计算与 AllToAll 通信的融合，**先计算后通信，
  沿 N 轴（末维）切分**。支持 MXFP8 / MXFP4 量化输入与原生 BF16 输入，输出统一为 BF16。
- 计算公式：每卡 `A_i` 的 shape 为 `(M, K)`（各卡行分片、互不相同），共享权重 `B` 的
  shape 为 `(K, N)`，`rankSize` 为 NPU 卡数，MX 量化按 `blockSize = 32` 分组。

  - 每卡 `i` 计算 `C_i = dequant(A_i) @ dequant(B)`，得到 `(M, N)` 全量结果；
  - 将 `C_i` 沿 N 维均匀分成 `rankSize` 份，`chunkN = N / rankSize`，
    `chunk[j] = C_i[:, j*chunkN : (j+1)*chunkN]` 发给 rank `j`；
  - AllToAll 后每卡 `j` 持有输出 `Y_j`，shape 为 `(rankSize*M, N/rankSize)`，
    其中行块 `[i*M : (i+1)*M]` 来自 rank `i`（行聚集 + 列分散）。
- 实现要点（单融合 AIC/AIV Kernel，同机与跨机同一路径）：
  - AIC 逐轮按“跨机 peer、同机 peer、本卡”的顺序计算，每轮 `H` 行：远端列块写本卡
    对称区 TX[轮]，本卡列块直写输出 Y；
  - 每个 AIC 完成自己负责的某个 peer 分片后发布该 `(round, peer)` 的独立完成票；
    对应 AIV 只等待这个 peer 的全部 AIC 完成票，不等待其它 peer、其它 rank，也不执行
    `aclshmemx_barrier_all_vec`；
  - 一个物理 AIV 对应一个 peer，经 UDMA（同节点）或 RoCE（跨节点）PUT 完整窗口到
    目标卡对称区 Y；本卡列块不重复搬运；
  - 每一轮使用独立的对称 TX 窗口，没有 READY/GET、消费 ACK 或槽复用；AIC 写入下一轮
    窗口时，AIV 可搬运上一轮，从而为后续 MatMul 与通信提供流水重叠。

## 通信组件适配

本示例以原版 `ascend950_quant_matmul_alltoall` 为基线，重组 AIC 计算代码并适配公共通信组件：

- AIC 参考 `ascend950_matmul_alltoall_mx_slice_n.hpp`，使用显式通信轮、目标 peer、tile 循环。
  本卡 Y 与远端 TX 使用相同的行布局，按 peer 选择输出基址，共用取 tile 和 MMAD 调用。
  AIC 只负责生产数据；每个 `(round, peer, AIC)` 通过独立的 64B ticket 通知 AIV，
  不使用旧的 lane 级 ticket/ACK 状态机，也不依赖整轮 CrossCoreFlag。

- 通信结构参考 PR #220 的 `ascend950_hierarchical_alltoall.hpp` 和
  `ascend950_mxfp8_alltoall_matmul_transport.h`：一个物理 AIV 直接对应一个 peer，
  构造 `HierarchicalCommSwizzle` 与公共七参数 `CommBlock`，每个融合轮调用一次 `CommBlock`
  搬运完整 peer 窗口。同节点走 UDMA，跨节点走 RDMA；本卡输出由 AIC 直写 Y，对应 AIV 不搬运。
- `CommBlock` 在内部按 `laneNum` 条数据 QP 切分当前窗口；本算子不再维护旧的 lane 级
  状态机。UDMA 的数据 QP 在全部轮次提交后统一 quiet；RDMA 由 `CommBlock` 在每次调用内完成 quiet。
- `TileHierarchicalReady` 只在本卡全部 PUT 完成后发布完成状态，不再承担远端 READY/GET 控制。
  网络完成通知在数据 QP quiet 之后复用 QP0；AIV 0 等待所有 source 的完成状态后才退出。
- 完整 TX 按 `[round][peer]` 寻址，各窗口互不覆盖。物理 AIV 编号等于 peer rank；无对应 peer 的 AIV
  不轮询、不提交数据。
- 保留 MXFP8/MXFP4/BF16 计算模板、轮粒度、精度比较、数据缓存与打点区间。
  通信打点表示 AIV 提交/peer 级等待区间，不等同于硬件链路的持续 busy 时间。

参考：[PR #220](https://gitcode.com/cann/catccos/pull/220/diffs)，本次对照提交
`2ab9a31e15cc8e32c7a44493e7fb9129f2bd4107`。
该 PR 已移除 `tile_remote_copy_hierarchical.hpp`，网络搬运收进 `CommBlock`，
MTE tile 使用 `tile_remote_copy.hpp`。本算子只使用 PR 提供的公共实现，不修改其他算子。

## 约束说明

- `rankSize`（NPU 卡数）：`2..32`；仅支持多卡 AllToAll，最多两节点。`rpn` 必须等于编译期的
  `CATCCOS_HIERARCHICAL_GROUP_SIZE`，单机也会使用该域划分（默认 `16`）。
- 启动要求物理 AIV 数量不少于 `rankSize`，AIC 数量不超过 64；host 在启动前检查。
- `N` 必须能被 `rankSize` 整除，且 `(N / rankSize) % 256 == 0`（tile 不跨 peer 边界）。
- `K` 必须是 `32` 的倍数（MX 量化按 `blockSize=32` 分组对齐）。
- 数据格式 `fmt`：`0`=MXFP8、`1`=MXFP4、`2`=原生 BF16；输出一律为 BF16。
- `lanes`（每个对端的数据 QP 数）：`1..4`。完成通知在全部数据 QP quiet 后复用 QP0，
  不额外占用控制 QP。
- 通信轮粒度 `QMMA_OPT_COMM_INTERVAL`：`1..64`，跨机两端必须一致。
- 输入 `A_i` 为 `(M, K)`，各卡不同；输入 `B` 为 `(K, N)`，各卡相同（共享权重）。

## 使用方式

### 1. 编译项目

进入示例目录并执行编译脚本（先加载 CANN 与 Python 环境）：

```bash
cd examples/ascend950_quant_matmul_alltoall

# HNS_1825 机器
SHMEM_RDMA_BACKEND=HNS_1825 bash scripts/build.sh

# XSCALE 机器
SHMEM_RDMA_BACKEND=XSCALE bash scripts/build.sh

# 8+8 跨机时，group size 必须与 QMMA_RPN 一起改成 8
SHMEM_RDMA_BACKEND=HNS_1825 bash scripts/build.sh \
  -DCATCCOS_HIERARCHICAL_GROUP_SIZE=8
```

- 可用 `QMMA_BUILD_DIR` 指定其他构建目录，默认为 `build/ascend950_quant_matmul_alltoall/`。
- 构建产物：`build/ascend950_quant_matmul_alltoall/bin/ascend950_quant_matmul_alltoall`。

### 2. 运行单机示例

```bash
# 单机双卡 MXFP8，默认 shape（逐行读取 scripts/test_shapes.csv）
bash scripts/run.sh 0,1

# 指定 fmt=2（BF16）与单组 shape，并设置通信轮粒度
QMMA_M=256 QMMA_K=64 QMMA_N=1024 QMMA_OPT_COMM_INTERVAL=3 \
  bash scripts/run.sh local 0,1 2
```

`run.sh` 依次生成输入与 golden、启动各 rank、逐 rank 比较输出，任一步失败返回非零；
各 rank 日志写入 `QMMA_DATA_DIR/logs/<run>/shape.../rank<N>.log`（终端同时实时显示）。

**参数说明**：

```text
run.sh [local] <device_list> [fmt] [slots] [lanes] [rpn] [world_size] [rank_start] [ip_port]
run.sh cross <node_id> <node0_ip> [local_device_start]
```

| 参数 | 说明 | 默认 |
| :-- | :-- | :-- |
| `device_list` | 本机参与运行的 NPU 编号列表，逗号分隔 | 必填 |
| `fmt` | `0`=MXFP8，`1`=MXFP4，`2`=BF16 | 0 |
| `slots` | 旧命令行兼容字段，当前实现忽略 | 2 |
| `lanes` | 每个 peer 的数据 QP 数，范围 `1..4`；完成通知复用 QP0 | 1 |
| `rpn` | 每节点 rank 数（node-major 编址），必须匹配编译期 group size | 16 |
| `world_size` | 总 rank 数，必须不小于 2 | 设备数 |
| `rank_start` | 本节点起始 rank id | 0 |
| `ip_port` | 跨机 bootstrap 地址（单机不使用） | - |

### 3. 环境变量

| 环境变量 | 说明 | 默认 |
| :-- | :-- | :-- |
| `QMMA_OPT_COMM_INTERVAL` | 通信轮粒度，每轮行数 = 值 × 128，范围 `1..64`；未设置时按 `M` 自适应 `clamp(⌈M/256⌉, 1, 32)`；跨机两端须一致 | 自适应 |
| `QMMA_M` / `QMMA_K` / `QMMA_N` | 单组 shape 的三个维度 | 读 CSV |
| `QMMA_TEST_SHAPES` | 自定义 shape CSV 路径 | `scripts/test_shapes.csv` |
| `QMMA_BUILD_DIR` | 构建目录 | `build/ascend950_quant_matmul_alltoall` |
| `QMMA_DATA_DIR` | 数据、缓存与日志根目录 | `out/qmma/functional` |
| `QMMA_LOG_DIR` | rank 日志根目录 | `QMMA_DATA_DIR/logs` |
| `QMMA_SHMEM_HEAP_MB` | 对称堆大小（MiB，须为偶数）；不设置时按 shape 自动计算 | 自动 |

### 4. 跨机运行

两端使用同一构建与相同 shape/fmt/lane/轮粒度，分别执行（`NODE0_IP` 为节点 0 地址）。
默认构建的 hierarchical group size 为 16，因此默认跨机用例是每节点 16 rank：

```bash
# 节点 0（16 卡示例）
export NODE0_IP=192.0.2.1
QMMA_RPN=16 QMMA_WORLD_SIZE=32 QMMA_LANES=2 QMMA_M=256 QMMA_N=8192 QMMA_K=64 \
  bash scripts/run.sh cross 0 "$NODE0_IP"

# 节点 1
QMMA_RPN=16 QMMA_WORLD_SIZE=32 QMMA_LANES=2 QMMA_M=256 QMMA_N=8192 QMMA_K=64 \
  bash scripts/run.sh cross 1 "$NODE0_IP"
```

- 位置参数：`run.sh cross <node_id> <node0_ip>`；
- `QMMA_RPN` 每节点 rank 数，必须等于 `CATCCOS_HIERARCHICAL_GROUP_SIZE`；`QMMA_WORLD_SIZE`
  为总 rank 数，`QMMA_LANES` 为数据 QP 数；
- 结果写入 `out/qmma/cross_node_P<P>/node<node_id>/`。
