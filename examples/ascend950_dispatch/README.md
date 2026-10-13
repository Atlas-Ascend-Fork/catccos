# Ascend950 Dispatch

## 功能说明

Ascend950 Dispatch 面向 MoE 专家并行（Expert Parallel，EP）场景，将本 rank 的 BF16 token
按行动态量化为 INT8，再根据专家路由发送到目标 rank，输出 INT8 token、反量化 scale 和路由元数据。
通信使用 SHMEM 对称内存与 MTE Put；计算专家选择的 gating/top-k 网络不属于本算子。

- 同一 token 命中同一个 rank 上的多个专家时，X 和 scale 只发送一份，top-k 索引及权重保留各专家槽位。
- 全局专家按 rank 连续、等量分布；接收时将全局专家索引转换为目标 rank 的本地专家索引。
- 接收结果按源 rank 升序排列，同一源 rank 内按原始 token 顺序排列，不按专家展开。
- 支持连续 rank 分组，通过 group-aware 调度分散目的访问、均分发送工作，不改变模型路由。
- 支持不同 rank 接收数量不相等、零接收和非整 tile 尾块；输出只包含有效行，不追加 padding token。
- 支持输出直接引用 SHMEM 接收区，或复制到调用者提供的独立 GM 输出缓冲。

本文描述 example 的 **BF16 输入 → 动态 INT8 量化 → INT8 通信 → INT8 输出**配置。
两个device配置均显式指定 `BlockQuantizeBf16Int8`，不根据输出类型自动选择FP8量化。
目标平台为 Ascend950；调用入口为本 example 的原生 C++ 程序。

## 参数说明

### Shape 符号

| 符号 | 含义 |
|---|---|
| `M` | 每 rank 输入 token 数，即本 example 的 BS / token per rank |
| `H` | 每个 token 的 hidden 维度；CSV 中的 `K` 列对应此维度 |
| `R` | EP rank 数，即 `rankSize` |
| `T` | 每 token 的 top-k 槽位数，即 `numTopk` |
| `E` | 专家总数；每 rank 部署 `E/R` 个专家 |
| `N` | 本 rank 实际接收 token 数，即 `rankPrefixMatrix[R-1, rankIdx]` |
| `O` | 输出缓冲的 token 容量，即 `numOutputTokens`；本 example 设置 `O=N` |
| `S` | 本 rank 的实际发送份数，即 `sum(numTokensPerRank)`，包含发给自身的份数 |

以下名称对应 `DeviceOp::Arguments`。数据指针在接口中以 `GM_ADDR` 传递，
表中数据类型表示指针指向的元素类型。二维张量采用连续行优先布局，GM 行间不插入 padding。

### 输入

| 参数名 | 输入/输出/属性 | 数据类型 | Shape / 格式 | 含义 |
|---|---|---|---|---|
| `ptrX` | 输入 | BFLOAT16 / `bfloat16_t` | `[M,H]`，RowMajor | 本 rank 的原始 token 特征；每行独立量化 |
| `ptrTopkIdx` | 输入 | INT64 / `int64_t` | `[M,T]`，RowMajor | 每个 token 的全局专家索引；有效索引范围为 `[0,E)` |
| `ptrTopkWeights` | 可选输入 | FLOAT32 / `float` | `[M,T]`，RowMajor | 各 top-k 槽位的权重；`hasTopkWeights=true` 时有效，本 example 启用；只随路由搬运，不乘入 X |
| `ptrNumTokensPerRank` | 输入 | INT32 / `int32_t` | `[R]` | 本 rank 发往各目标 rank 的 token 数，同一 token→rank 去重后计数 |
| `ptrIsTokenInRank` | 输入 | UINT8 / `uint8_t` | `[M,R]`，RowMajor | 非零表示该 token 需要发送到对应 rank；生成器使用0/1 |
| `ptrRankPrefixMatrix` | 输入 | INT32 / `int32_t` | `[R,R]`，RowMajor | 行为源 rank，列为目标 rank；`[s,d]` 表示源 rank 0至s发往d的包含式累积和，所有 rank 使用相同矩阵 |

路由张量和计数由调用者准备，必须相互一致；算子不在设备侧从 `ptrTopkIdx` 自动生成 mask 或全局 prefix。
本 example 由 Python 预生成二进制文件。BF16 输入文件以 `uint16_t` 保存原始位模式，
device 按 `bfloat16_t` 解释，不做数值类型转换。

### 输出

| 参数名 | 输入/输出/属性 | 数据类型 | Shape / 格式 | 含义 |
|---|---|---|---|---|
| `ptrRecvX` | 输出 | INT8 / `int8_t` | `[O,H]`，RowMajor | 接收的量化 token，前N行有效 |
| `ptrRecvScales` | 输出 | FLOAT32 / `float` | `[O]` | 每行一个反量化 scale，近似恢复公式为 `x = recv_x * recv_scale` |
| `ptrRecvTopkIdx` | 输出 | INT64 / `int64_t` | `[O,T]`，RowMajor | 命中本 rank 的槽位转换为本地专家索引；其他槽位为-1 |
| `ptrRecvTopkWeights` | 可选输出 | FLOAT32 / `float` | `[O,T]`，RowMajor | 权重功能启用时，保留命中本 rank 的权重；其他槽位为0 |
| `ptrRecvSrcIdx` | 输出 | INT32 / `int32_t` | `[O]` | token 在其源 rank 输入中的行号，不是全局 token ID；源 rank 由接收分段和 prefix 确定 |

有效输出行数N取决于路由，各 rank 不要求相同。若 `O>N`，尾部不填充、不拷贝。
同 rank 多专家命中会保留多个有效 top-k 槽位，但只占一行 `recv_x`。

### 属性与调度参数

| 参数名 | 类别 | 类型 | 含义与约束 |
|---|---|---|---|
| `gemmShape` | 属性 | `Catlass::GemmCoord` | 取 `{M,1,H}`，用于描述 token 矩阵，不表示本算子执行 GEMM |
| `commBlockShape` | 属性 | `Catlass::MatrixCoord` | 取 `{commBlockM,H}`；限制通信块及 UB tile 的行数 |
| `rankIdx` | 属性 | `uint32_t` | 本 rank 编号，范围 `[0,R)` |
| `rankSize` | 属性 | `uint32_t` | EP通信域大小，本 example 范围 `[1,128]`；复用项目的 `MAX_RANK_SIZE` 限制 UB 路由表容量 |
| `groupSize` | 属性 | `uint32_t` | 连续分组的卡数，范围 `[1,R]`；尾组可以不足该数 |
| `commInterval` | 属性 | `uint32_t` | 正整数；重排路径每个目标每轮窗口最多包含 `commInterval × ceil(AIV数/R)` 个通信块，且不超过M行；Direct路径不使用该参数调度 |
| `numTopk` | 属性 | `uint32_t` | T；本 example 要求 `1 <= T <= min(R,128)` |
| `numExperts` | 属性 | `uint32_t` | E；本 example 取 `R × expertsPerRank`，`expertsPerRank>0` |
| `numOutputTokens` | 属性 | `uint32_t` | O；输出容量，调用者应保证能容纳N行，本 example 取N |
| `hasTopkWeights` | 属性 | `bool` | 是否搬运 top-k 权重；本 example 固定为true |

### 通信缓冲与中间数据

| 参数名 | 类别 | 数据类型 / 单位 | 有效 Shape / 含义 |
|---|---|---|---|
| `ptrSymmetric` | 通信缓冲 | 字节寻址的 SHMEM 对称内存 | 包含接收区、可选重排区及量化区；同一 shape 下各 rank 的区域大小与偏移一致 |
| `ptrQuantX` | 中间输出 | INT8 / `int8_t` | `[M,H]`；量化后的本地 token，供后续搬运消费 |
| `ptrQuantScales` | 中间输出 | FLOAT32 / `float` | `[M]`；每个输入 token 一个反量化 scale |
| `symmetricRecvXOffset`、`symmetricRecvTopkIdxOffset`、`symmetricRecvTopkWeightsOffset`、`symmetricRecvSrcIdxOffset`、`symmetricRecvScalesOffset` | 布局属性 | `uint64_t`，字节 | 五类接收字段相对于 `ptrSymmetric` 的起始偏移 |
| `symmetricStageXOffset`、`symmetricStageTopkIdxOffset`、`symmetricStageTopkWeightsOffset`、`symmetricStageSrcIdxOffset`、`symmetricStageScalesOffset` | 布局属性 | `uint64_t`，字节 | 五类本地重排字段相对于 `ptrSymmetric` 的起始偏移；Direct路径不使用 |

`ptrQuantX` 和 `ptrQuantScales` 由调用者分配，不得与输入、接收或重排有效区域重叠。
本 example 将它们放在 SHMEM 分配尾部；五个输出指针直接引用接收区，因此无需末尾整张量复制。
`GetWorkspaceSize()` 返回0，表示没有额外 workspace，不表示算子不需要上述缓冲。

## 约束说明

- 所有 rank 使用相同的M、H、T、E和分组/调度配置，并选择相同 kernel；本 example 不提供各 rank 不同输入M的运行模式。
- `M>0`、`H>0`、`commBlockM>0`；`commBlockM` 可以大于M，尾块按实际行数裁剪。
- `R*M` 不超过 `uint32_t` 上限；路由计数及 prefix 还必须可由 INT32 表示。SHMEM 申请量须在可用堆容量以内。
- H不要求32 B对齐。量化 UB 需求为 `2048 + 11*roundUp(H,128) + 32` 字节，
  必须不超过 `Catlass::Arch::Ascend950::UB_SIZE`（248 KiB）；以 `BlockQuantizeBf16Int8::CanImplement(H)` 为准。
- 输入仅支持有限 BF16 值。全零行输出全零INT8和scale=0；非零行的 `127/max(abs(x))` 必须能由FP32表示。
- 专家索引、mask、每目标计数和全局 prefix 必须描述同一份路由；容量预留不会自动修复不一致的路由。
- group调度只改变访问顺序和工作分配，不能消除目标rank/group热点，也不保证所有shape都能满核。
- 同一通信轮次不能混用Direct与重排kernel；它们的同步阶段不同。

## 数据布局与内存申请

### 接收和重排布局

令 `D[s,d]` 为源rank s发往目标rank d的去重token数，则：

```text
rankPrefixMatrix[s,d] = sum(D[0..s,d])
recvBegin(s,d) = s == 0 ? 0 : rankPrefixMatrix[s-1,d]
recvEnd(s,d)   = rankPrefixMatrix[s,d]
N[d]          = rankPrefixMatrix[R-1,d]

stageBegin(d) = sum(numTokensPerRank[0..d-1])
stageEnd(d)   = stageBegin(d) + numTokensPerRank[d]
```

源rank s写入目标d的 `[recvBegin,recvEnd)` 行，每个源拥有不重叠区间，区间内保留原始token顺序。
重排区按目标rank分段。X、scale、top-k索引、top-k权重和srcIdx均独立存放，不交错打包。

每个源的发送份数满足 `S <= M*T`；单个目标的接收上限为 `R*M`，
并不受 `M*T` 限制。输出N是实际路由计数，不是固定shape推导出的均衡值。

### SHMEM 容量

`main.cpp` 初始化的每rank SHMEM堆容量为1 GB（1024³ B），再按shape上限一次申请算子缓冲。
定义 `roundUp(x,32)` 为向上取整到32的倍数：

| 区域 | 预留容量 | 每行预算 |
|---|---|---|
| 接收区 | `recvCapacity = roundUp(R*M,32)` | X、scale、top-k索引/权重、srcIdx，共五类字段 |
| 重排区 | Direct为0；否则 `stageCapacity = roundUp(M*T,32)` | X、scale、top-k索引/权重、srcIdx，共五类字段 |
| 量化区 | `quantCapacity = roundUp(M,32)` | INT8 X和FP32 scale |

五类字段的单行预算与总字节数为：

```text
rowBytes = H*sizeof(ElementToken)
         + T*(sizeof(ElementTopkIdx) + sizeof(ElementTopkW))
         + sizeof(ElementIdx) + sizeof(ElementScale)

symmetricBytes = (recvCapacity + stageCapacity)*rowBytes
               + quantCapacity*(H*sizeof(ElementToken) + sizeof(ElementScale))
```

容量按32行取整，使各字段区域起始地址满足32 B对齐；这只是区域尾部预留，不增加有效输出行数、
GM行步长或通信量。UB二维tile的行步长另行按32 B对齐，补齐字节不写入GM。
例如INT8、H=129时，UB行步长可为160 B，而GM行步长仍为129 B。

EP16、H3584、T8、M256时走Direct：每行预算3688 B，每rank申请16,024,576 B，约15.282 MB。
这里的申请量与初始化的1 GB堆容量是两个概念。

## 执行流程与分层

`main.cpp` 根据M选择两种共享输入输出接口的kernel：

| 条件 | Kernel / Device配置 | 执行流程 |
|---|---|---|
| `M<512` | `Ascend950DispatchDirect` / `Ascend950DispatchDirectConfig::Device` | 量化 → 发布量化结果 → UB收集并直写目标接收区 → 发布远端写 → Complete |
| `M>=512` | `Ascend950Dispatch` / `Ascend950DispatchConfig::Device` | 量化 → 发布量化结果 → GM重排 → 发布重排结果 → 分块AllToAllV → 发布远端写 → Complete |

两条路径在量化后都使用 `aclshmem_fence + AscendC::SyncAll<true>()` 完成本卡AIV同步，
远端写入后使用 `aclshmem_fence + aclshmemx_barrier_all_vec()` 完成跨rank同步。
重排加AllToAllV路径还在本地GM重排完成后使用
`aclshmem_fence + AscendC::SyncAll<true>()` 同步本卡AIV，再开始发送。
Direct没有GM重排区；两条路径的 `Complete` 在输出别名到接收区时跳过复制，独立输出缓冲则复制有效行。
复用同一接收区时，调用者需保证各rank上一轮的输出复制或后续消费已完成，再开始下一轮写入；
当前example的输出直接别名到接收区，连续性能测试不执行额外输出复制或中间消费。
512是example的算法选择阈值，不代表所有EP和拓扑下已经验证的性能交点。

### 分核和分块

- `blockNum` 取 `PlatformAscendCManager::GetInstance()->GetCoreNumAic()`，AIC侧不执行计算。
  全部AIV按token步进完成量化，每token只量化一次。
- Direct发送、重排和AllToAllV以 `GetBlockIdx()` 为worker编号，
  `GetBlockNum()*GetSubBlockNum()` 为worker数W；全部AIV参与量化和必要的同步，实际发送任务数可以小于W。
- 重排将S条有效发送行均分，worker w负责 `[floor(S*w/W),floor(S*(w+1)/W))`。
  Direct按group交错的目标顺序分配目标内连续行任务：非空目标至少一个任务，
  重目标可由多个worker处理，但每行只有一个writer。
- AllToAllV将每目标每轮的窗口与单个通信块分开：窗口最多包含
  `commInterval*ceil(W/R)` 个 `commBlockM` 行块，且不超过M行。任务索引取余R选择目标、
  除以R选择行块，worker按W步长领取任务，轮次之间没有额外barrier。
- `commBlockM` 限制二维tile行数，实际tile还受字段列数和UB容量约束；增大它或 `commInterval` 不保证一定提速。

GM→UB和UB→GM均直接复用CATLASS的 `CopyGm2Ub` / `CopyUb2Gm`，
按 `TileM × TileN` 及对应layout搬运，保留非对齐hidden、UB行padding和尾块的精确写入。
不额外补齐通信数据；AllToAllV的 `TileRemoteCopy` 路径保持不变。

直发和重排共用的block将输入中连续、且属于当前目标rank和worker区间的token，
通过一次CATLASS `CopyGm2Ub` 加载到现有二维X tile中；不连续的token仍分别收集。
预加载不跨越tile容量或worker区间，元数据仍按原token顺序处理，
`FlushX`仍在达到原tile行数或目标区间结束时发送，不增加发送门槛或同步。
路由、top-k索引、权重和scale保持按需从GM读取，不额外分配输入元数据缓存。

### 实现组件

| 组件 | 职责 |
|---|---|
| `Ascend950DispatchConfig` / `Ascend950DispatchDirectConfig` | 配置字段类型、RowMajor layout、tile拷贝和kernel/device类型 |
| `Ascend950Dispatch` / `Ascend950DispatchDirect` | 参数转换、阶段编排及同步 |
| `BlockCommSchedulerDispatch` | group-aware目的顺序、通信轮次和块任务调度 |
| `CommBlockDispatch` | UB缓冲、token收集、元数据转换、分块搬运和输出复制 |
| `CopyGm2Ub` / `CopyUb2Gm` / `TileRemoteCopy<Put,Mte>` | GM↔UB直接复用CATLASS，远端tile搬运复用CATCCOS |
| `BlockQuantizeBf16Int8` | 独立的per-token动态量化接口 |

字段模板参数依次为 `ElementToken=int8_t`、`ElementScale=float`、
`ElementTopkIdx=int64_t`、`ElementTopkW=float`、`ElementIdx=int32_t`。
`ElementToken` 表示通信/输出类型，原始输入为BF16。

量化接口位于 `include/catccos/comm/block/block_quantize.hpp`：

```cpp
BlockQuantizeBf16Int8 quantize;
quantize(resource, input, output, scales, rows, hidden, aicoreIdx, aicoreNum);
```

量化先计算FP32的 `amax=max(abs(x))`、`multiplier=127/amax` 和
`scale=1/multiplier`，再按FP32→INT32（RINT）→FP16（ROUND，反量化系数1）→INT8（TRUNC）转换。
接口使用调用者的UB和事件ID0，不内置跨核同步；返回后完成本worker写回，其他worker消费前需建立同步。

## 构建与精度测试

以下命令均在 **CATCCOS仓库根目录**执行。准备Ascend950的CANN/Bisheng、SHMEM及Python 3环境；
构建脚本调用仓库 `examples/utils/setup.sh` 配置依赖。

```bash
bash examples/ascend950_dispatch/scripts/build.sh

# 8卡功能测试
bash examples/ascend950_dispatch/scripts/run.sh 0,1,2,3,4,5,6,7

# 16卡功能测试
bash examples/ascend950_dispatch/scripts/run.sh \
  0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

`run.sh` 逐行读取 `scripts/test_shapes.csv`，调用 `get_data.py` 生成输入/golden，
启动各rank的原生程序，再由 `verify.py` 校验五类输出。
数据写入 `examples/ascend950_dispatch/out/`，下一shape会替换该目录中的二进制测试数据。

### 自定义 shape 与校验

`scripts/test_shapes.csv` 的列如下；修改或增加行即可调整功能测试shape：

| 列 | 含义 |
|---|---|
| `M` | 每rank输入token数 |
| `K` | hidden维度H |
| `group_size` | 每组rank数 |
| `top_k` | 每token专家槽位数T |
| `experts_per_rank` | 每rank专家数，总专家数为R乘以该值 |
| `comm_block_m` | 通信块/UB tile的最大行数 |
| `comm_interval` | 重排路径每轮窗口的块数 |
| `route_mode` | 可选：`balanced`（默认）、`duplicate`（同token多专家命中同rank）、`skewed`（目标集中在首组，其他rank零接收） |

功能shape覆盖BS=1、2、4、8、12、16、20、24、28、32，以及非对齐hidden、511/512/513算法切换边界和非均衡路由。
`recv_x`、`recv_topk_idx`、`recv_topk_weights`、`recv_src_idx` 按位校验；
`recv_x_scales` 使用FP32相对误差 `1e-5`、绝对误差0，全零输入的scale必须为0。

生成器采用确定性路由：

```text
dstRank     = (srcRank + tokenIdx + topkSlot) % R
localExpert = (3*tokenIdx + topkSlot) % expertsPerRank
expertId    = dstRank*expertsPerRank + localExpert
```

这是默认 `balanced` 模式：在T≤R约束下，一个token命中T个不同rank；全局累加后，每rank恰好接收M*T行。
`duplicate` 模式把相邻top-k槽位路由到同一rank，验证去重发送；`skewed` 模式只命中首组rank，验证热点与零接收。各模式的mask、计数、prefix和golden按同一份路由生成。

原生程序参数顺序为：

```text
ascend950_dispatch rank_size rank_id ip_port num_tokens hidden group_size top_k
                   experts_per_rank comm_interval comm_block_m data_path
                   [device_id_list] [warmup_iterations performance_iterations]
```

`get_data.py` 参数顺序为 `rank_size num_tokens hidden group_size top_k experts_per_rank data_dir`；
`verify.py` 参数为 `data_dir rank`。功能和性能脚本负责组织这些调用，C++不生成测试数据。

## 性能测试

### 测试场景

`scripts/run_perf.py` 使用 `msprof` 采集本example的原生二进制，不调用其他框架的MoE算子。

| 配置 | 取值 |
|---|---|
| 机器 / EP | 单机，rank数由 `--ep` 指定，默认16；需要同样数量的可用设备 |
| group | `--group-size` 指定每组rank数，默认4；必须整除EP，设备顺序应与实际拓扑匹配 |
| 专家数 / topK | 共384个专家，每rank自动分配 `384 / EP` 个；topK=8 |
| hidden / 数据类型 | H=3584；BF16输入，INT8通信和输出 |
| token per rank | 默认1、2、4、8、16、24、32、48、64、96、128、192、256，共13组；可用 `--tprs` 指定 |
| kernel | 默认13组均满足M<512，使用Direct；自定义M仍由main中的512阈值选择kernel |
| 默认迭代 | 100次，无额外warmup |
| 默认通信配置 | `comm_block_m=2`、`comm_interval=1`；Direct不使用后者调度 |

性能输入复用 `get_data.py` 的均衡路由。输入生成不计时，性能运行不执行精度校验，应先通过功能测试。
默认性能shape集中定义在 `run_perf.py` 的 `TOKENS`，与功能测试CSV独立，且不绑定EP。
当前固定topK=8、专家数384，要求 `8 <= EP <= 128` 且EP整除384。
`--experts-per-rank` 可显式传入，但必须满足 `EP * experts_per_rank = 384`。

### 运行和汇总

以下命令均在CATCCOS仓库根目录执行，结果保存在项目内的 `./tmp/` 下。
先完成构建，在运行终端加载匹配的CANN环境及SHMEM环境，确认 `msprof` 在PATH中：

```bash
source 3rdparty/shmem/install/set_env.sh

python3 examples/ascend950_dispatch/scripts/run_perf.py run \
  --ep 16 \
  --output ./tmp/dispatch_perf_run01
```

脚本自动启动EP个rank，不需要host配置文件。未传 `--devices` 时，使用设备 `0,...,EP-1`；
传入时必须包含EP个不重复设备。每连续 `group_size` 个逻辑rank应对应一个真实group。
初始化地址默认为 `tcp://127.0.0.1:27029`，`--master` 可修改地址/端口。
`--iters` 必须至少100；`--comm-block-m` 可调整tile行数；`--timeout` 默认每shape 1800秒。

例如EP12、3个group、每rank32个专家，测试默认13组shape：

```bash
python3 examples/ascend950_dispatch/scripts/run_perf.py run \
  --ep 12 \
  --devices 0,1,2,3,4,5,6,7,8,9,10,11 \
  --group-size 4 \
  --comm-block-m 2 \
  --output ./tmp/dispatch_perf_ep12_run01
```

设备列表仅为示例，须替换为12张健康卡，按3个完整真实group排列。
添加 `--tprs 256` 可单测BS=256；EP8同理使用 `--ep 8`，每rank自动分配48个专家。
这里只调整测试参数，不需要修改或重新编译kernel。

输出目录下的 `node_0` 必须不存在；重复运行请使用新的输出目录，避免混入已有trace。
每个shape保存 `rank_*.log`、profiler trace及 `samples_rank_*.json`，完成后生成
`./tmp/dispatch_perf_run01/performance.csv`；上述EP12命令的结果为
`./tmp/dispatch_perf_ep12_run01/performance.csv`。

从保存的采样结果重新汇总：

```bash
# EP16
python3 examples/ascend950_dispatch/scripts/run_perf.py summarize \
  ./tmp/dispatch_perf_run01 --output ./tmp/dispatch_perf_run01/performance.csv

# EP12
python3 examples/ascend950_dispatch/scripts/run_perf.py summarize \
  ./tmp/dispatch_perf_ep12_run01 --output ./tmp/dispatch_perf_ep12_run01/performance.csv
```

汇总从采样JSON读取实际EP和本次选择的shape，不需要再次指定 `--ep`。
可接收多个收集目录，但不能包含重复rank/shape，必须覆盖本次全部shape及rank `0,...,EP-1`。
不同EP、设备映射或通信参数的结果不能混合汇总；旧版EP16采样JSON仍可按默认13组shape汇总。
`--binary` 可指定原生程序路径，默认 `build/bin/ascend950_dispatch`。

### 采样口径

1. 读取 `op_summary*.csv` 的 `Task Duration(us)`，按名称/类型匹配dispatch任务，
   排除 `AI_CPU`、`N/A` 和 `HCCL` 记录；不以 `MIX_AIV` 作为唯一筛选条件。
2. 每rank必须有且仅有一个匹配的导出文件，任务数必须恰好等于iters；取最后100次耗时。
   默认iters=100时包含首次执行；例如iters=110时只统计最后100次。
3. 每个rank将100次耗时升序排序，取最快70次求算术平均；再将EP个rank的平均值求平均，作为最终平均耗时。
   p50/p95/p99对EP个rank各自选出的70次样本（共 `EP * 70` 个）按线性插值计算。
   这是各rank独立去除30次慢样本的口径，不代表整轮Dispatch的完成时延，也不是各rank的耗时求和。
4. 耗时覆盖选中kernel的量化、路由/收集、通信及同步；重排路径还包含GM重排。
   输出使用独立缓冲时，kernel内的Complete复制也在计时范围内。
5. 不包含host数据生成、SHMEM初始化或编译时间；没有在迭代间插入额外all_reduce或计算负载。
   不同测试框架的缓存和前置负载条件需要单独对齐。

`op_summary*.csv` 支持逗号或Tab分隔。若提示 `No op_summary*.csv found`，
请检查该shape的 `rank_*.log` 和对应 `rank_*` 目录中的profiler导出；
若提示 `got []`，说明找到导出文件但没有匹配记录，应检查表头、实际算子名及 `--op-pattern`，
不能仅凭此错误判断算子执行失败。

如名称不同，可用 `--op-pattern` 指定正则表达式。缺少CSV、任务数不符或采样无效时脚本报错，
不会回退到主机计时；若msprof未自动导出CSV，应先按所用工具版本完成导出。
定位运行错误时检查对应 `node_0/tprs_<M>/rank_<r>.log`。

### 结果字段与带宽计算

表头保持下列顺序。单位约定：**KB=1024 B，MB=1024² B，GB=1024³ B**。

| 数据项 | 单位 / 含义 |
|---|---|
| OP | `CATCCOS Dispatch BF16-INT8` |
| token per rank | 每rank输入token数M |
| EP | `--ep` 指定的实际rank数 |
| p50_us | EP个rank各自最快70次样本汇总后的50%分位数，μs |
| p95_us | EP个rank各自最快70次样本汇总后的95%分位数，μs |
| p99_us | EP个rank各自最快70次样本汇总后的99%分位数，μs |
| 数据量KB | 每rank逻辑INT8载荷 `M*3584*8*1 / 1024` |
| 平均耗时 | 每rank最快70次的均值，再对EP个rank均值求平均，μs |
| 算法带宽 | `数据量B * 1,000,000 / 平均耗时 / 1024³`，GB/s |
| 带宽利用率 | `算法带宽 / 196 * 100`，百分数；不强制截断到100% |

BS=32时，逻辑载荷为917,504 B，即896 KB。`1,000,000` 仅用于微秒转秒。
196基准按二进制GB/s解释；如果采用十进制硬件标称带宽，需要先转换单位。

逻辑载荷包含本地路由、不乘EP，不包含scale、top-k索引/权重、srcIdx及控制开销。
因此“带宽利用率”是**约定逻辑载荷相对196 GB/s的比值**，不是跨group协议带宽利用率，
也不是物理链路占用率；196参考值不随EP自动调整，不能将不同EP的数据换算为EP16性能。
真实路由有同rank多专家命中时，应按去重发送计数另算实际流量。
比较不同实现时，除shape外还需对齐路由、去重语义、量化类型、输出布局和计时范围。

## 文件索引

| 文件 | 职责 |
|---|---|
| `main.cpp` | 加载输入、SHMEM分配、算法选择、启动和输出保存 |
| `ascend950_dispatch_host.h` | 类型定义、shape检查、二进制读写和设备内存管理 |
| `ascend950_dispatch_device.h` | 两个kernel的device配置 |
| `scripts/build.sh` / `scripts/run.sh` | 构建及多rank精度测试 |
| `scripts/get_data.py` | 输入与golden生成，含CPU量化参考 |
| `scripts/verify.py` | 五类输出精度校验 |
| `scripts/run_perf.py` | 性能采集与CSV汇总 |
| `scripts/test_shapes.csv` | 功能测试shape列表 |

核心实现位于 `include/catccos/comm/kernel/ascend950_dispatch.hpp`、
`ascend950_dispatch_direct.hpp`，以及 `include/catccos/comm/block/` 下的
`comm_block_dispatch.hpp`、`comm_block_scheduler_dispatch.hpp` 和 `block_quantize.hpp`。
