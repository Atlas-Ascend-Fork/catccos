# Ascend950 Dispatch FFN Combine 示例

本示例在 Ascend950 上完成 MoE 的 routing、dispatch、两段 FFN、combine 和 unpermute。算子输入输出为 BF16，FFN 的两段矩阵乘使用 E4M3 数据与 E8M0 block scale。

主要执行流程如下：

```text
BF16 tokens
  -> moe_init_routing_v2
  -> per-token MXFP8 quant + packed dispatch
  -> MXFP8 GMM1
  -> SwiGLU + dynamic MXFP8 quant
  -> MXFP8 GMM2
  -> BF16 combine
  -> moe_token_unpermute
  -> BF16 output
```

## 目录结构

```text
ascend950_dispatch_ffn_combine/
├── CMakeLists.txt
├── README.md
├── ascend950_dispatch_ffn_combine_device.h
├── ascend950_dispatch_ffn_combine_host.h
├── main.cpp
└── scripts/
    ├── build.sh
    ├── run.sh
    └── test_shapes.csv
```

## 环境准备

- Ascend950 NPU，驱动与 CANN 环境可用。
- 仓库依赖已初始化，CATLASS 与 ACLSHMEM 已完成构建和安装。
- `3rdparty/shmem/install/set_env.sh` 存在且可正常加载。
- Python 3 已安装 PyTorch、NumPy 和 SciPy；安装 `torch_npu` 时使用 NPU 生成 golden，否则自动回退到 CPU。

## 编译

在仓库根目录执行：

```bash
cd examples/ascend950_dispatch_ffn_combine
bash scripts/build.sh
```

可执行文件生成在：

```text
build/bin/ascend950_dispatch_ffn_combine
```

## 运行与精度校验

```bash
bash scripts/run.sh <device_id_list> [expert_num]
```

参数说明：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `device_id_list` | 逗号分隔的 NPU 编号，同时决定 rank 数量 | 必填 |
| `expert_num` | 全局 expert 数，必须能被 rank 数整除 | `16` |

例如在 0、1 两张卡上运行：

```bash
bash scripts/run.sh 0,1 16
```

脚本会依次读取 `scripts/test_shapes.csv`，生成输入和 golden 数据，为每张卡启动一个 rank，并校验各 rank 输出。所有用例都应在日志中显示 `PASS`。

运行参数可通过环境变量调整：

| 环境变量 | 说明 | 默认值 |
|---|---|---|
| `TOPK` | 每个 token 选择的 expert 数 | `6` |
| `GMM1_M_TILE` | GMM1 的 M 方向 tile，可选 `128` 或 `256` | `128` |
| `ROUTING_MODE` | 路由数据模式，可选 `random` 或 `balanced` | `random` |
| `CSV_FILE` | 自定义 shape CSV 文件 | `scripts/test_shapes.csv` |
| `EXEC_BIN` | 自定义可执行文件路径 | `build/bin/ascend950_dispatch_ffn_combine` |
| `IPPORT` | ACLSHMEM 通信地址 | `tcp://127.0.0.1:27008` |

示例：使用均衡路由和 M tile 256 运行四卡：

```bash
ROUTING_MODE=balanced GMM1_M_TILE=256 bash scripts/run.sh 0,1,2,3 16
```

## 配置测试 shape

`scripts/test_shapes.csv` 使用以下格式：

```csv
M,K,N
4,7168,4096
64,7168,4096
256,7168,1024
512,7168,4096
```

当前 MXFP8 实现要求：

- `M`、`K`、`N` 均为正整数。
- `K` 能被 `256` 整除。
- `N` 能被 `512` 整除。
- `expert_num >= TOPK`，且 `expert_num` 能被 rank 数整除。
- `device_id_list` 最多包含 8 张卡。

## 输出

测试数据与结果写入当前示例的 `output/` 目录。主要文件包括：

- `output_<rank>.bin`：算子输出。
- `out_gather_combine_<rank>.bin`：golden 输出。
- `verify_<rank>.log`：逐 rank 精度校验日志。

任一 rank 执行失败、输出文件不完整或精度校验未出现 `PASS` 时，`run.sh` 会返回非零状态。

## PyTorch eager 接入

在已安装 CANN 和 torch_npu 的环境下，从仓库根目录构建 A5 binding。

```bash
source examples/utils/setup.sh -soc_type Ascend950
cmake -S . -B build-torch -DCATLASS_BISHENG_ARCH=a5 -DCATCCOS_TORCH_EXTENSION=ON   -DCATCCOS_BUILD_TESTS=OFF -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build-torch --target catccos_torch -j8
export LD_LIBRARY_PATH="$PWD/build-torch/lib:$LD_LIBRARY_PATH"
```

每个 rank 使用一个进程，先选择 NPU，再按相同顺序调用 init、forward 和 finalize。

```python
import torch
import torch_npu

torch.npu.set_device(device_id)
torch.ops.load_library("build-torch/lib/libcatccos_torch.so")
torch.ops.catccos.init(rank, world_size, 2 * 1024**3, "tcp://host:port")
y = torch.ops.catccos.ascend950_dispatch_ffn_combine(x, expert_idx, gate_weight, w1, w1_scale, w2, w2_scale)
torch.ops.catccos.finalize()
```

所有张量须位于 init 选定的 NPU，具有连续 stride 和 `ACL_FORMAT_ND` 存储。
`x` 和 `y` 是 BF16 `[M,K]`，全局专家 ID（int32，每行互异）和 gate（float32）为 `[M,TopK]`。
`w1`、`w2` 是 E4M3 `[localExperts,N,K]`、`[localExperts,K,N/2]`，
对应 E8M0 scale 为 `[localExperts,N,K/32]`、`[localExperts,K,N/64]`。
M>0，K 是 256 的倍数，N 是 512 的倍数。同一进程顺序调用，各 rank 协调 finalize。
