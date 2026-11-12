# Ascend950 分层 AllToAll

源 rank `r` 的 `input[p]` Put 到 rank `p` 的 `output[r]`。每个 peer 的数据是等量的
`Element` RowMajor 矩阵，并传输对应的 `ElementScale` scale 矩阵。单机和跨机使用同一 kernel：self 走 MTE，同机走 UDMA，跨机走 RDMA。

`main.cpp` 统一设置 `Element = float8_e4m3_t`、`ElementScale = float8_e8m0_t`，
Host operator 和 Device Config 均接收这两个类型模板参数。每个 peer 的数据形状为 `m × n`，
要求 `m/n` 为正数且 `n` 能被 32 整除；scale 按每行 32 个元素一组组织，形状为 `m × (n/32)`。
默认类型下，每个 peer 的有效通信字节数为 `m*n + m*(n/32)`，两种张量分别使用对齐槽。
Kernel 的 `Params` 根据 `m/n` 和元素类型计算 scale 列数及两个 peer stride，入口无需传入这些派生参数。
本例只验证数据和 scale 的原样传输，不执行量化或矩阵乘法。

## 代码结构

沿用 `ascend950_mxfp8_matmul_alltoall` 的 Options → Config/DeviceOp → Host operator 结构：

| 层 | 文件 | 职责 |
|---|---|---|
| 入口 | `main.cpp` | 参数、固定 tiling、DeviceOp 组装和单次功能测试 |
| Host | `ascend950_hierarchical_alltoall_host.h` | ACL/SHMEM 生命周期、布局、分配、二进制读写和路由检查 |
| Config | `ascend950_hierarchical_alltoall_device.h` | 组装 Scheduler、Tile、Block、Kernel、Device 类型 |
| Scheduler | `include/catccos/comm/block/comm_block_hierarchical_swizzle.hpp` | peer、transport、通信轮次和尾轮 |
| Block | `include/catccos/comm/block/comm_block_hierarchical_copy.hpp` | 链路选择、UB stage、MTE TileSwizzle 切块、网络 QP 分片与提交 |
| Tile | `include/catccos/comm/tile/tile_remote_copy.hpp` | 复用原有 MTE 特化执行 GM → UB → GM 搬运 |
| Ready | `include/catccos/comm/tile/tile_hierarchical_ready.hpp` | 完成等待、按来源通知、接收就绪汇总 |
| Kernel | `include/catccos/comm/kernel/ascend950_hierarchical_alltoall.hpp` | 地址选片、commIdx 循环、接收就绪协议 |
| Barrier | `include/catccos/comm/kernel/ascend950_hierarchical_barrier.hpp` | CATCCOS 分层同步，同组 UDMA、跨组 RDMA |

Host operator 继承 `CatccosOperator`，实现分配、结果写出、参数检查、workspace 和 kernel 类型接口。
`CocTilingParams.m/n` 表示每个 peer 的矩阵行列数，`COMM_BLOCK_M` 表示 MTE 每轮搬运行数，在 Device Config 中固定为 64 并作为 Swizzle 模板参数传入；
纯通信参数通过 `CheckCocTilingParams` 检查，不使用基类的 GEMM 公共参数检查。
`KernelParams.ptrA/ptrC` 分别指向数据输入和输出，`customPtrs[0/1]` 指向 scale 输入和输出。
四个张量均为 Host operator 持有的同一块 SHMEM 分配中的内存视图，
由其析构函数统一释放，不调用使用 `aclrtFree` 的 `FreeDeviceSpace`。
`scripts/gen_data.py` 在 CPU 上使用 PyTorch 生成 E4M3 数据及 E8M0 scale 位编码，
Host 用类型化缓冲区读取二进制输入，执行后写出结果，`scripts/verify_result.py` 对照 golden 文件逐字节校验。
kernel 通过模板接收 Scheduler，具体分层策略由 Config 选择。
Config 为数据和 scale 分别传入 `TileRemoteCopy<..., CopyDirect::Put, CopyTransport::Mte>`，
并传入 CATLASS 的 `EpilogueIdentityTileSwizzle` 作为 MTE 的 TileSwizzle；
分层通信逻辑统一放在 `comm_block_hierarchical_copy.hpp`，不再额外包装一个分层 Copy Tile。

## 通信约束

- `CORE_COUNT=32` 是策略支持的 AIV/peer 上限，不是自动读取的芯片总核数。
  一个 AIV 对应一个 peer，负责该 peer 的所有 QP；增加 QP 不增加 AIV 数。MIX 1:2 的 blockDim 由平台接口 `GetCoreNumAic()` 获取，
  超出 peer 数量的 AIV 直接退出。AIV 0 发送完成后兼任本卡接收协调者，AIC 0 验证就绪等待，其他 AIC 空闲。
- Swizzle 使用 `HierarchicalCommSwizzle<CORE_COUNT, GROUP_SIZE, COMM_BLOCK_M>`，组规模和 MTE 通信块行数均为编译期参数，example 默认 `GROUP_SIZE=16`。
  运行参数 `group_size` 必须与编译值一致，并对应真实服务器；UBX launcher 要求它等于本机参与的卡数。
  四卡 UB 直连 clique 不等于整台服务器的 UDMA 域；单机 16 卡必须 group=16。
- MTE 每个 `commIdx` 对应 `COMM_BLOCK_M=64` 行，尾轮截断，`coreLoops=ceil(m/64)`。
  UDMA/RDMA 对每个 peer 的数据和 scale 分别只调度一轮，非空张量的 `coreLoops=1`，直接发送整张量。
  `QP_NUM=1` 时每个连续张量在单请求上限内只提交一次 NBI；多 QP 将整张量分成连续、不重叠的片段，
  分片边界相对整张量按 32 字节对齐，
  不足 32 字节的尾部归最后一个 QP，小尺寸可能只用到部分 QP。超出 256 MiB 单请求上限时再分段。
  行 padding 布局逐行分片，只传输有效列，不覆盖 padding。
  MTE 在 Block 内用 `TileSwizzle` 切成二维 tile，保留源/目标原始 stride 后调用原有 MTE Tile。
  tile 的列数最多为 `32768 / sizeof(Element)`，行数受 `tileM * AlignUp(tileN * sizeof(Element), 32) <= 32768` 约束，
  并截断到当前块的实际行列数。数据和 scale 分别计算，连续布局和行 padding 布局共用该路径。
  例如 FP8 的 `64 × 256` 块调用一次 `64 × 256` Tile，`64 × 1024` 块切成两个 `32 × 1024` Tile。
  网络路径仍由 Block 直接调用 SHMEM QP 接口，不经过 MTE TileSwizzle。
- 同一个 AIV 先传输该 peer 的数据，再传输 scale，两者使用相同 peer/QP，并顺序复用 UB 和事件。
  UDMA 的 Block 只提交 NBI；RDMA 在每个张量的 Block 调用内逐 QP quiet，MTE 由 Block 等待搬运事件。
  数据和 scale 都提交后，Ready Tile 再确保所有 QP 完成，才通过 QP0 发送本批次 ready 标记并等待该标记发送完成。
  两种网络传输都先提交一个张量的各 QP 分片，再按上述位置等待完成。
- 每个 peer 槽按 32 字节对齐；scale 槽尾部不足 32 字节时填充 `0xa5`。数据和 scale 均独立校验有效内容和未写 padding。
- SHMEM 按 `QP_NUM=1/2/4` 配置每个 peer 的 UDMA/RDMA QP 数，默认 1。
  每个同组远端 peer 的所有已配置 QP 都必须有 UDMA route，缺失即报错。
  明确关闭 shared jetty，保证不同 AIV 不会并发修改同一共享 SQ/CQ。
- Host 顺序：读取/上传/清空 → 输入 barrier → 数据 kernel（含本卡接收就绪等待）/stream 同步 → 写出结果 → 输出结束 barrier。
  本机所有进程退出后，运行脚本校验二进制结果。数据 kernel 后不再额外调用完成 barrier。
  输入和输出结束 barrier 仍使用 CATCCOS 自己的单 AIV kernel、QP0 和独立的对称标志区，
  与数据 kernel 在同一 stream 上顺序执行，负责开始和释放资源前的全局同步。

## 接收就绪与融合接入

`TileHierarchicalReady` 使用独立的对称 workspace，每个来源的 ready、每个目的 peer 的发送值、
本卡 frontier 均占独立缓存行。Host 用 `aclshmem_calloc` 初始化一次，并传给 `DeviceOp.Initialize`。
当前功能样例将数据和 scale 的完整 peer 张量作为一个批次，`epoch=1`：

```text
每个发送 AIV：数据 + scale → Notify(peer, transport, epoch, qpNum, scratch, eventId)
本卡 AIV 0：先完成自己的发送 → WaitAll(epoch) → 发布本卡 frontier
本卡 AIC 0：WaitReady(epoch) → 返回（功能样例不执行 matmul）
```

`Notify` 在 UDMA/RDMA 的所有数据 QP quiet 后，用 QP0 写接收卡的 `ready[source]`；
self 路径在 MTE3 完成后写本地槽。`WaitAll` 等待每个来源的值都不小于本批次 epoch，
再发布本卡 frontier。`WaitReady` 可供融合算子的 AIC/AIV 在读取本批次数据前调用。
这保证本卡收齐所有来源，**不是所有卡同时结束这一批次的全局 barrier**。

多批次融合由调用方保留外层窗口循环：按本批次实际行数重新构造 Swizzle，更新数据和 scale
的源/目的偏移，完成该 peer 的所有 BlockCopy 后调用一次 `Notify`。每个窗口使用递增的非零 epoch；
同一个 peer 的发送者必须按 epoch 顺序提交。协调者依次 `WaitAll(epoch)`，消费者计算前 `WaitReady(epoch)`。
可以像本例让 AIV 0 先发送再协调，也可以由融合 kernel 分配独立的协调 AIV（须保证实际核数足够）。
调用者为通知提供独占且空闲的 `UB_BYTES=256` 字节 UB 和 event，保留每个 peer 的单 AIV/QP 所有权。

跨 kernel 复用 workspace 也必须递增 `Arguments.epoch`，不得回绕、运行中清零或并发启动共享 QP 的 kernel。
各窗口的接收数据须保留到消费完成；ready 不包含消费完成确认，使用循环缓冲区时需要额外的释放协议。
本例仍是单批次功能测试，不包含多批次 matmul 或性能测试。

## 构建

SHMEM 固定为 `v1.8.0` 发布节点 `ed07fa687d1ca30e636b5a86e1e6ceba82d88535`。
SHMEM 使用该节点的原始代码，CATCCOS 不应用任何子仓补丁。
升级必须同时重编 SHMEM 和示例，不能只切换源码继续用旧 install。
`setup.sh` 会复用有效的 `SHMEM_HOME_PATH` 或已有 install，重跑示例的 `build.sh` 不会自动升级该 SDK。

HNS_1825 下无需手工提供 `/etc/hccl_rootinfo.json`：SHMEM 仍从驱动获取 UDMA EID/拓扑，
从 `/var/run/ascend-topologyd/virtualTopology.xml` 获取 RoCE 网卡 IP。
v1.8.0 已修复服务器标识的获取：优先使用驱动返回的 super-pod/server 标识，
不可用时回退到 `/proc/sys/kernel/random/boot_id`；使用回退路径时，同机容器须看到一致的宿主机 boot ID。
本例直接使用 SHMEM 的自动拓扑生成，不再覆盖 `get_server_id` 或依赖其私有头文件。
已有 rootinfo 仍遵循 SHMEM 的文件优先规则。
`GROUP_SIZE` 只控制本例的同组/跨组通信选择，仍须与实际服务器边界一致。

需要目标机 Linux、Ascend950、匹配的 CANN/Bisheng 和 RDMA provider。
网卡、驱动及 `IBV_EXTEND_DRIVERS` 配置参考
[SHMEM RDMA 环境要求](https://gitcode.com/cann/shmem/blob/v1.8.0/examples/rdma_demo/README.md)。
以下使用 HNS_1825 后端，在 CATCCOS 仓库根目录执行：

```bash
export CANN_PATH=/usr/local/Ascend/ascend-toolkit  # 按实际安装路径修改
source "${CANN_PATH}/set_env.sh"
export SHMEM_RDMA_BACKEND=HNS_1825
export SHMEM_HNS_1825_PI_ON_CHIP=1  # 与 SDK 和目标固件一致；legacy 模式设为 0

cd examples/ascend950_hierarchical_alltoall
GROUP_SIZE=16 SHMEM_BUILD_ARGS="-enable_rdma -rdma_backend HNS_1825 -multi_link" bash scripts/build.sh
```

`-multi_link` 用于跨 UB clique 的多级交换网络选路。XSCALE 机器须改用相应后端及 provider。
构建不再探测 SDK commit/SOC/backend 或检查动态重定位，使用者须保证 SDK 与源码及配置匹配。
构建成功后保存 `version.info`；运行时仍拒绝与构建快照不一致的 SDK。

HNS1825 的 legacy/PI-on-chip 必须匹配真实 NIC 固件。构建前必须显式设置
`SHMEM_HNS_1825_PI_ON_CHIP=0|1`，不再自动探测。该变量控制 CATCCOS kernel，
SHMEM 库自身也必须按相同模式构建，其 CMake 选项为 `ACLSHMEMI_HNS_1825_PI_ON_CHIP=0|1`。

该 example 注册在公共 examples 列表中，无需独立构建开关，编译架构仍为 `a5`。
请使用 `scripts/build.sh` 校验配置值、传入后端/固件编译参数并保存 SDK 版本快照。
切换 XSCALE 后端时，设置 `SHMEM_RDMA_BACKEND=XSCALE` 并 `unset SHMEM_HNS_1825_PI_ON_CHIP`。
`build.sh` 会强制重编头文件生成的设备库，修改头文件后务必重跑。

## 运行

以下命令均在 `examples/ascend950_hierarchical_alltoall` 目录执行。
每个 rank 对应一个进程和一张 NPU，`run.sh` 的唯一位置参数是本机设备列表；
当前支持连续的 `0,1,...,N-1`，每台机器参与的卡数必须相同，总 rank 数不超过 32。
两端的 `PYTHON` 解释器需要安装支持 `torch.float8_e4m3fn` 的 PyTorch，保持两端 PyTorch 版本一致；
数据生成使用 CPU，不需要 `torch_npu`。默认 `PYTHON=python3`，可用 `export PYTHON=/实际环境/bin/python3` 指定。

### 单机运行

```bash
NODE_RANK=0 NODE_COUNT=1 GROUP_SIZE=16 IPPORT='tcp://127.0.0.1:8788' \
    bash scripts/run.sh 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

### 双机 32 卡

跨机启动方式参考 [SHMEM RDMA Multi-QP Demo](https://gitcode.com/cann/shmem/blob/v1.8.0/examples/rdma_qp_demo/README.md)。
两台服务器各使用 16 张 NPU，分别设置 `NODE_RANK=0` 和 `NODE_RANK=1`。
两端完成构建，并使用相同的源码版本、RDMA 后端、CSV 用例及 QP 数。

| 服务器 | NODE_RANK | 本机 NPU | 全局 rank |
|---|---|---|---|
| A | 0 | 0～15 | 0～15 |
| B | 1 | 0～15 | 16～31 |

在两端终端中设置 `RANK0_IP` 为服务器 A 的实际 IP，再分别执行以下命令。
两端 `NODE_COUNT`、`GROUP_SIZE`、`IPPORT`、`QP_NUM` 必须一致，端口 `8788` 应空闲且两端可达。

```bash
# 服务器 A
NODE_RANK=0 NODE_COUNT=2 GROUP_SIZE=16 IPPORT="tcp://${RANK0_IP:?请设置服务器 A 的 IP}:8788" \
    bash scripts/run.sh 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

```bash
# 服务器 B
NODE_RANK=1 NODE_COUNT=2 GROUP_SIZE=16 IPPORT="tcp://${RANK0_IP:?请设置服务器 A 的 IP}:8788" \
    bash scripts/run.sh 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

两端在 SHMEM 初始化超时前完成启动。每个 rank 对本卡使用 MTE，对 15 个同机 peer 使用 UDMA，
对 16 个跨机 peer 使用 RDMA；默认每个 peer 使用 qp0。
调整参与卡数时，两端先用相同的 `GROUP_SIZE` 重编，例如 `GROUP_SIZE=2 bash scripts/build.sh`；
运行时同时修改 `GROUP_SIZE` 和设备列表，保持编译值、运行值与本机参与卡数一致。

### 多 QP direct 功能验证

本例使用 direct：直接发送到目标 peer，不经过 relay。保持上面的构建命令不变，
两端启动时设置相同的 `QP_NUM=1`、`2` 或 `4`；无需为每个 QP 数重新编译。
下面以 QP=4、双机 32 卡为例，`RANK0_IP` 仍为服务器 A 的实际 IP：

```bash
# 服务器 A
QP_NUM=4 NODE_RANK=0 NODE_COUNT=2 GROUP_SIZE=16 \
    IPPORT="tcp://${RANK0_IP:?请设置服务器 A 的 IP}:8792" \
    bash scripts/run.sh 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

```bash
# 服务器 B
QP_NUM=4 NODE_RANK=1 NODE_COUNT=2 GROUP_SIZE=16 \
    IPPORT="tcp://${RANK0_IP:?请设置服务器 A 的 IP}:8792" \
    bash scripts/run.sh 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

先验证 QP=1 基线，再把两端同时改为 QP=2、QP=4；前一轮两端都退出后再启动下一轮。
默认 CSV 覆盖 MTE 的单轮搬运、m 方向尾轮和多轮搬运，以及 UDMA/RDMA 的整张量发送；每个用例执行一次 kernel，并逐元素校验接收数据、scale 及各自的 padding。
`QP_NUM` 也适用于直接执行二进制，位置参数不变。两端必须使用相同 QP 数。
多 QP 功能需要在目标 NPU 上验证；增加 QP 并不保证提高带宽。

### 功能测试结果

默认 CSV 包含三个用例。以下列出服务器 B 的启动信息及用例汇总：

```text
node=1/2 world=32 group=16 qp_num=1 logs=.../<时间戳>-node1-qp1
[PASS] m=64 n=64
[PASS] m=257 n=256
[PASS] m=1024 n=512
```

每个用例中，各 rank 在执行 kernel 前打印一次总 rank 数及各通信方式的卡数，包含本卡：

```text
[transport] rank=16 world=32 mte=1 udma=15 rdma=16 qp_num=1
```

MTE 卡数包含本卡，UDMA 为同组其他卡，RDMA 为跨组卡，三者之和等于 `world`。
该行保存在 rank 日志中，`run.sh` 汇总日志时也会输出到终端。

逐 rank 的校验结果 `PASS rank=N` 保存在 `LOG_DIR/world32_m<m>_n<n>/rankN.log`。
`run.sh` 等待本机所有进程，检查退出码和 PASS；**两端所有用例通过且脚本均返回 0** 才算完整通过。
任一 rank 失败时，脚本打印本机各 rank 日志并停止后续用例，避免继续复用失败的 bootstrap 会话。

### 运行参数

| 环境变量 | 默认值 | 含义 |
|---|---|---|
| `NODE_RANK` / `NODE_COUNT` | 0 / 1 | 本机编号 / 服务器数 |
| `GROUP_SIZE` | 构建时 16；运行时本机卡数 | 连续 rank 组规模，编译值与运行值必须一致 |
| `QP_NUM` | 1 | 每个网络 peer 的 QP 数，支持 1、2、4；所有 rank 必须一致 |
| `IPPORT` | `tcp://127.0.0.1:8788` | bootstrap 地址；跨机必须为 rank 0 真实地址 |
| `CSV_FILE` | `scripts/test_shapes.csv` | `m,n`，每个 peer 的矩阵行列数，m/n 为正数且 n 是 32 的倍数 |
| `PYTHON` | `python3` | 生成和校验文件使用的 Python 解释器 |
| `LOG_DIR` | `scripts/logs/<时间戳>-node<N>-qp<Q>` | 每个 case 的 rank 日志 |

二进制参数：`rank_size rank_id ip_port m n group_size data_path [device_id_list]`。
`run.sh` 自动为各用例生成本机 rank 的文件并传入 `data_path`，原有脚本运行命令不变。
数据保存在 `LOG_DIR/world<world>_m<m>_n<n>/data`：

| 文件 | 内容 |
|---|---|
| `rank_<r>_a.bin` / `rank_<r>_a_scale.bin` | rank r 发给各 peer 的数据和 scale |
| `golden_<r>.bin` / `golden_scale_<r>.bin` | rank r 应收到的各源 rank 数据和 scale |
| `output_<r>.bin` / `output_scale_<r>.bin` | 算子实际接收结果 |

文件按 peer 升序排列，每个槽包含有效元素和对齐 padding。
生成器按 `(source, destination)` 使用固定种子，golden 按 AllToAll 置换生成；
两台机器只生成各自本地 rank 的文件，不需要共享目录或复制文件。
当前生成器对应 `main.cpp` 的 E4M3/E8M0 默认类型；更换元素类型时须同步修改文件生成方式。
每个用例执行一次 kernel，数据、scale 和 padding 全部逐字节一致才打印 `PASS rank=N`。

## 验证与排错

在目标机上执行构建和算子功能验证：

```bash
bash scripts/build.sh
GROUP_SIZE=16 bash scripts/run.sh 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

`run.sh` 按 `scripts/test_shapes.csv` 启动算子，`verify_result.py` 对接收数据、scale 及各自的 padding 做逐字节比较，
并检查每个 rank 的退出码与 PASS。

初始化失败时，`aclshmemx_init_attr failed: <code>` 只是外层返回值，需要查看此前的 SHMEM 内部错误。
本例通过 SHMEM 公开日志回调将错误直接写入 stderr 并立即刷新，`run.sh` 会将它们保存在
各 rank 日志中，不依赖 SHMEM 独立日志目录或 `SHMEM_LOG_TO_STDOUT`。
默认只打印 ERROR/FATAL；需要初始化过程时，在两端原启动命令前设置：

```bash
export SHMEM_LOG_LEVEL=INFO
```

| 现象 | 优先检查 |
|---|---|
| `invalid ip_port` | 两端地址必须是完整的 `tcp://IP:端口`，端口后不能带多余字符 |
| `Failed to connect to ...` | rank 0 是否正常启动并监听、两端 IPPORT 是否一致、TCP 是否可达 |
| `aclshmemx_init_attr failed` | rank 日志中最早的 `[shmem]` 错误，以及本机 XML、驱动拓扑和实际加载的 SDK |
