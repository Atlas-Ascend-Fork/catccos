# `ascend950_alltoall_matmul_fused` 示例

在Ascend 950PR&950DT系列产品上按 M 窗口重叠 AllToAll 通信与 MatMul 计算。AIV 负责搬运和就绪通知，AIC 在窗口数据齐备后计算，使用 FP32 累加并输出 BF16。

通过运行时的 `DTYPE` 选择格式。默认 `mxfp8`，也可设置 `DTYPE=bf16`。两种格式使用同一套构建和启动脚本。

## MXFP8 输入与约束

设 `rankSize` 为总卡数，`M` 为每卡输出行数，`N` 为输出列数，`K` 为拼接后的完整 K，`localK = K / rankSize`。每卡输入 A 有 `rankSize × M` 行。

| 张量 | 每卡形状 | 类型与布局 |
| --- | --- | --- |
| A | `[rankSize × M, localK]` | FP8 E4M3，RowMajor |
| B | `[K, N]` | FP8 E4M3，ColumnMajor |
| AScale | `[rankSize × M, localK/64, 2]` | E8M0，紧凑排列，每 32 个 K 元素共用一个 scale |
| BScale | `[N, K/64, 2]` | E8M0，每 32 个 K 元素共用一个 scale |
| C | `[M, N]` | BF16，RowMajor |

支持单机或双机，每机 1/2/4/8/16 卡；双机参与卡数须相同，总卡数不超过 32。编译时通过 `GROUP_SIZE` 指定每机参与卡数，运行时设备列表长度必须与其一致。`M` 支持 1～300000，目标性能形状为 10000～300000。

K、N 在运行时指定，默认 `K=rankSize×256、N=4096`。`localK` 须为 64 的正整数倍，`N` 须为 16 的正整数倍。L1 K tile 自动选择能整除 localK 的 256/128/64；N 支持不足 256 列的尾块。AScale 等价于 `[rankSize, M, localK/32]`，调用方无需补齐行数。SHMEM 接收区仍在内部对齐各来源槽位，不影响紧凑的输出 `C[M,N]`。旧版带行填充的 AScale 文件需重新生成，`data.json` 版本检查会拒绝复用旧数据。

通信使用本卡 MTE、同机 UDMA、跨机 RDMA，由 `HierarchicalCommSwizzle` 选择路径并划分每窗的通信块。A、AScale 分别遍历通信块，再由 `TileHierarchicalReady::Notify` 通知接收端。每个 peer 由一个 AIV 负责，网络通信使用 QP0。RDMA 依赖 HNS_1825 单 QP 保序，每窗连续提交 A、AScale、ready，全部窗口提交后只 quiet 一次；每窗 ready 发送值预先初始化并保持只读，所有 peer 共用。UDMA/MTE 仍等待数据完成后发布 ready。独立协调 AIV 调用 `WaitAll` 推进 frontier，AIC 通过 `WaitReady` 等待对应窗口。ready 使用公共模板的 32 位计数和 64 B 槽位，只在所有卡上一轮结束后清零，固定发送值不清零。通信窗口默认 512 行，MTE 通信块为 64 行，MatMul 的 M tile 固定为 256 行。

## 构建

两台机器使用相同代码和依赖版本，先加载本机 CANN、RDMA provider 环境。当前已验证环境为 CANN `9.2.T101`、RDMA 后端 `HNS_1825`。
数据生成和校验需要 Python 3.10+ 与 NumPy。

从仓库根目录执行，后续命令均在示例目录中运行：

```bash
source /path/to/cann/set_env.sh  # 替换为本机安装路径
export SHMEM_RDMA_BACKEND=HNS_1825
export SHMEM_HNS_1825_PI_ON_CHIP=1
export IBV_EXTEND_DRIVERS=/path/to/libhrn5-rdmav34.so  # 替换为本机 provider 路径
cd examples/ascend950_alltoall_matmul_fused
GROUP_SIZE=16 bash scripts/build.sh
```

构建产物位于仓库的 `build/bin/a2a_matmul` 和 `build/bin/liba2a_matmul_impl.so`。已有 SHMEM SDK 必须与当前源码及后端配置匹配；首次构建会准备缺失的依赖。

`GROUP_SIZE` 是编译期参数，默认 16；例如每机 8 卡时，两机均用 `GROUP_SIZE=8 bash scripts/build.sh` 重新编译，再用 `LOCAL_RANKS=8` 运行。仅修改运行环境变量不会改变已编译的通信分组。

HNS doorbell 模式须与 SHMEM SDK 和目标固件一致：PI-on-chip 设置 `SHMEM_HNS_1825_PI_ON_CHIP=1`，旧固件的 legacy 模式设置为 `0`。

## 运行

### 双机 32 卡

默认每机使用物理卡 0～15。两机分别生成相同输入，并设置同一个 rank 0 地址：

```bash
export IPPORT=tcp://192.0.2.1:12560  # 替换为第一台机器的可访问地址
python3 scripts/gen_data.py generate /tmp/mxfp8_data --world 32 --m 10000
```

随后在对应机器上分别启动：

```bash
# 第一台机器
NODE_RANK=0 DATA_PATH=/tmp/mxfp8_data ITERATIONS=3 bash scripts/run.sh

# 第二台机器
NODE_RANK=1 DATA_PATH=/tmp/mxfp8_data ITERATIONS=3 bash scripts/run.sh
```

两端均返回 PASS 才表示本次检查通过。脚本会检查所选 NPU 是否空闲；SHMEM 默认可自动生成拓扑，也支持环境已有的 `/etc/hccl_rootinfo.json`。

### 单机 16 卡

```bash
NODE_COUNT=1 IPPORT=tcp://127.0.0.1:12560 \
  M=10000 ITERATIONS=3 bash scripts/run.sh
```

### 常用参数

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `NODE_COUNT` / `NODE_RANK` | `2` / `0` | 机器数量 / 本机编号 |
| `LOCAL_RANKS` / `START_DEVICE` | `16` / `0` | 本机卡数须等于编译时的 `GROUP_SIZE` / 起始物理卡号 |
| `M` | `10000` | 每卡输出行数，两端一致 |
| `K` / `N` | `WORLD_SIZE×256` / `4096` | 完整 K / 输出列数，两端一致 |
| `WINDOW_ROWS` | `512` | 通信窗口，可选 256/512/1024，两端一致 |
| `ITERATIONS` | `20` | 复用同一份输入的调用次数 |
| `DATA_PATH` | 自动生成 | 输入和输出数据目录 |
| `CHECK_RESULT` | `1` | 最后一次调用结束后校验输出，设为 0 跳过 |
| `LOG_DIR` | 示例目录下 `out/` 的独立子目录 | 日志及打点目录 |

`scripts/test_shapes.csv` 记录五个 32 卡测试形状，列顺序为 `DTYPE,M,K,N`。
MXFP8 使用 `(10000,4096,8192)` 和 `(300000,4096,8192)`，BF16 使用 `(3667,3328,12288)`、
`(4518,4096,12288)` 和 `(17,4096,12288)`，括号内按 `(M,N,K)` 顺序书写。

设置好各节点的 `NODE_RANK` 和相同的 `IPPORT` 后，两台机器执行同一条命令。脚本按 `DTYPE` 筛选形状：

```bash
DTYPE=mxfp8 CSV_FILE=scripts/test_shapes.csv bash scripts/run.sh
DTYPE=bf16  CSV_FILE=scripts/test_shapes.csv bash scripts/run.sh
```

旧格式的 `M,K,N` CSV 也可用于当前 `DTYPE`。只跑一个 shape 时直接传 `M`、`N`、`K`。
二进制参数为 `rank_size rank_id ip_port M N K [data_path] [device_id_list]`，由程序内部换算输入行数和 localK。

完整参数见 `bash scripts/run.sh --help`。

## 精度校验

`scripts/gen_data.py` 生成确定性的周期输入；各次调用复用同一份 A/AScale 和 B/BScale。最后一次调用结束后，检查每个 256 行计算 tile 的首尾行及全部列，参考值由 Python 独立计算并舍入到 BF16。输出文件按 BF16 保存，每个元素 2 字节。

文件读取、输入 H2D 和结果校验在计时外；每次调用的普通 GM → SHMEM 输入拷贝计入算子耗时。`CHECK_RESULT=0` 可跳过结果校验。

## 日志与设备打点

`LOG_DIR` 中包含各 rank 日志、`correctness.log`、运行配置及退出码。`run.sh` 结束时打印每轮本机最慢 rank 的主机调用耗时，以及去掉首轮后的中位数，并保存到 `summary.txt`。

唯一算子耗时 `total_ms`：上一轮隔离同步返回后开始，包含本轮 ready/frontier 重置、A/AScale 拷入 SHMEM、准备同步，以及融合执行到 `aclrtSynchronizeStream` 成功返回。一次性初始化、文件加载和输入 H2D、上一轮隔离同步、输出回读校验及打点导出不计入。

设备打点默认关闭。两机均使用以下选项重新构建，再按前面的运行方式执行：

```bash
GROUP_SIZE=16 bash scripts/build.sh -DENABLE_TIMER=ON
```

默认仅为最后一次调用开启打点并导出到 `LOG_DIR/trace/`，可用 `TRACE_ITERATION` 指定轮次；其他调用不执行阶段打点，日志 `TRACE_MODE` 显示每轮是否开启。将两机 trace 目录收集到同一台机器后转换，例如 32 卡、M=10000、窗口 512：

```bash
python3 scripts/trace.py node0/trace node1/trace \
  --world 32 --group-size 16 --m 10000 --window-rows 512 --output trace_report
```

输出包含 Chrome tracing JSON 和 `summary.csv`，用于查看窗口就绪、AIC 等待及计算范围。各卡时间以自己的 kernel 入口为零点，未做跨卡时钟校准。

阶段打点使用通用 `Tik/Tok`：`AIC` 记录计算 tile，`AIV` 记录窗口发送（包括每窗口的 quiet），`AIC_WAIT` 记录等待数据就绪，`AIV_WAIT` 记录协调核等待窗口到齐。`Tik/Tok` 自带核内 `PIPE_ALL` 同步，打点可能影响流水重叠，阶段耗时不能直接当作硬件利用率。


## BF16 输入与运行

BF16 的 A、B、C 都是 BF16，MatMul 使用 FP32 累加。A 按目标 rank 连续存放，B 逻辑上是列主序的 `[K,N]`，C 是行主序的 `[M,N]`。通信仅搬 A。`K` 须能被总卡数整除，`N` 须为 16 的倍数。和 MXFP8 一样，BF16 的 `LOCAL_RANKS` 须与编译时的 `GROUP_SIZE` 一致。默认 shape 为 `(3667,3328,12288)`，另两组验证 shape 为 `(4518,4096,12288)` 和 `(17,4096,12288)`，顺序为 `(M,N,K)`。

两种格式共用原 kernel 头文件 `ascend950_alltoall_matmul_fused.hpp` 内的窗口调度和输出 tile 划分，各自绑定对应的 MatMul block。每次通信前先清零 ready 和 frontier，再用跨 rank 屏障确认清零完成，随后按窗口从 1 编号。

跨机验证使用 SHMEM v1.8.0 的原生拓扑发现。沿用上面的构建方法，在两台机器使用相同 SDK 和后端配置。双机运行时先在节点 0 执行以下命令，节点 1 在看到 `BOOTSTRAP_READY` 后用相同参数启动，并将 `NODE_RANK` 改为 `1`。

```bash
DTYPE=bf16 NODE_RANK=0 IPPORT=tcp://<rank0-host>:12560 \
  M=3667 N=3328 K=12288 bash scripts/run.sh
```

默认运行一次，脚本生成可复现的本地输入，用 CPU 的 FP32 MatMul 转 BF16 参考值校验首轮每个输出元素。可用 `SEED` 更换输入。设置 `ITERATIONS` 后，后续轮次复用输入计时，首轮输出保留供校验。

## 计时与遮盖度

MXFP8 和 BF16 都支持 `OVERLAP=1`。在两台机器设置相同的 `IPPORT`，并分别设置本机的 `NODE_RANK`，
然后按格式选择一条命令：

```bash
DTYPE=mxfp8 OVERLAP=1 M=10000 N=4096 K=8192 bash scripts/run.sh
DTYPE=bf16 OVERLAP=1 M=17 N=4096 K=12288 bash scripts/run.sh
```

编译时使用 `ENABLE_TIMER=OFF`。默认预热 5 轮、测量 20 轮，`WARMUP` 为预热轮数，`ITERATIONS` 为总轮数。

AscendC 分别实测融合、仅通信、仅 MatMul 和串行四种模式。融合、仅通信和串行各计一次 ready 清零、
GM 到 SHMEM 拷贝和启动前跨 rank 屏障。仅 MatMul 从接收区 A 已就绪开始计时。串行模式在同一 stream 上
依次启动通信和 MatMul。仅 MatMul 和串行的首轮输出与融合输出逐元素比较，MXFP8 使用相同的抽样行，
BF16 比较全量输出。

脚本通过 `scripts/trace.py overlap` 汇总全部 rank 的原始样本。每轮取所有 rank 最大耗时，再求各轮平均值。
rank 0 将原始样本和汇总写入 `LOG_DIR/overlap.json`。MXFP8 使用带 Gloo 的 PyTorch 在主机侧汇总计时。
BF16 还在 NPU 上实测 HCCL、Torch MatMul 及 `HCCL → 接收布局转换 → Torch MatMul` 的串行链路，
`PYTHON_BIN` 须指向装有匹配 PyTorch 和 torch_npu 的环境。启动 HCCL 前脚本会再次检查设备占用。

```text
own_overlap_percent = (T_own_serial - T_fused) / min(T_own_comm, T_own_matmul) * 100
hccl_torch_overlap_percent = (T_hccl_torch_serial - T_fused) / min(T_hccl, T_torch) * 100
```

两种对照都使用实测串行时间，HCCL＋Torch 对照还包含不同通信实现和框架的开销。
查看 BF16 设备打点时，给 `scripts/trace.py` 加 `--dtype bf16` 并传实际的 `M`、`N`、`K`。
