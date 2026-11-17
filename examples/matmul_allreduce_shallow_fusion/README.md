# MatmulAllReduceShallowFusion

本示例通过主干 `OperatorRegistry` / `CatccosOperator` 和 SHMEM 调用迁移模板。支持Atlas A2系列产品、FP16、RowMajor A，以及 RowMajor/ColumnMajor B（transB=0/1）。

在仓库根目录、已初始化 conda 的 Bash 中执行。编译依赖及工具版本要求见[仓库环境准备](../../README.md)。Python 环境需安装相互兼容的 `torch`、`torch_npu`、`numpy`、`scipy`；双标杆生成会实际调用 NPU。当前验证环境为 CANN 9.1.1、PyTorch/torch_npu 2.12.0、`zystorch` 环境。

`setup.sh` 会初始化仓库子模块，并在需要时构建和加载 SHMEM；首次运行需要能访问子模块源。若已经有与本仓库子模块版本匹配的 SHMEM 安装，可先加载其 `install/set_env.sh`，公共 setup 会复用它。以下 CANN 默认路径可通过 `CANN_INSTALL_PATH` 改为本机安装目录。

```bash
source "${CANN_INSTALL_PATH:-/usr/local/Ascend/ascend-toolkit}/set_env.sh"
conda activate zystorch
source examples/utils/setup.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -c 'import torch, torch_npu, numpy, scipy'
bash examples/matmul_allreduce_shallow_fusion/scripts/build.sh /tmp/catccos-build
run_root=$(mktemp -d /tmp/matmul_allreduce_shallow_fusion.XXXXXX)
CATCCOS_SEED=2026 bash examples/matmul_allreduce_shallow_fusion/scripts/run.sh 0,2 3 257 255 \
    /tmp/catccos-build/bin "$run_root/case"
```

`0,2` 是两张设备的示例，请替换为本机可用且支持 SHMEM 互访的设备 ID。构建目录首次使用时应为空或不存在，后续可由本分支的四个样例共享；切换源码目录、CANN/SHMEM 安装，或切换到Ascend 950PR&950DT系列产品后应另用新目录，避免沿用旧 CMake 缓存。本例面向Atlas A2系列产品，编译目标为 `dav-c220`。运行时保留完整 `bin` 目录，其中包含可执行文件及其依赖的 `libmatmul_allreduce_shallow_fusion_impl.so`，并在运行终端加载上述环境。

CSV 批量回归（读取本样例的 [scripts/test_shapes.csv](scripts/test_shapes.csv)）：

```bash
bash examples/matmul_allreduce_shallow_fusion/scripts/run.sh 0,2 /tmp/catccos-build/bin
```

CSV 调用格式为 `run.sh device_id_list [bin_dir [new_output_dir]]`。`bin_dir` 默认是 `/tmp/catccos-build/bin`，与 `build.sh` 的默认构建目录一致；不指定输出目录时自动创建临时目录，并打印路径。每行结果放在 `case-<序号>/` 下，任何一行运行或精度校验失败都会使整批返回非零退出码。指定的输出目录必须尚不存在。

CSV 前三列沿用其他样例的 `M,K,N` 顺序，后续列记录 `transB`、`seed`。当前用例覆盖对齐尺寸和非对齐尾块，以及 transB=0/1。CSV 回归支持 2、4、8、16 卡。固定的 seed=2026 使同一 Python/PyTorch 环境下的数据可复现，避免每次重新抽取随机输入引起精度判定波动。修改 shape、种子或工具链后需要重新验证，不能仅凭 shape 保证任意随机输入都通过。

单个自定义 shape 的脚本参数：

```text
run.sh device_id_list M N K bin_dir new_output_dir [transB]
```

自定义模式仍默认使用随机输入；可以在命令前设置 `CATCCOS_SEED=2026` 复现输入。该设置通过公共生成器的可选 `--seed` 参数生效，CSV 模式使用各行记录的种子。`new_output_dir` 必须尚不存在。

二进制参数：

```text
matmul_allreduce_shallow_fusion rank_size rank_id ip_port M N K data_path [device_id_list [transB]]
```

B 文件在 transB=0/1 时分别按 K×N / N×K 存储。输入使用公共 `rank_<id>_a.bin`、`rank_<id>_b.bin`，各 rank 的 A/B 与现有 `gen_data.py` 默认行为一致。各 rank 输出为 `rank_<id>_output.bin`。标杆使用公共的矩阵乘法后归约流程。

`run.sh` 直接调用 `gen_data.py mmar --double-golden` 生成 CPU FP32 与 ACLNN FP16 双标杆，启动各 rank 执行一次，然后调用公共 `verify_result.py --golden_low`。精度公式和阈值沿用公共实现。单个 case 的所有数据位于其输出目录的 `output/` 下，生成日志为 `gen-data.log`，运行日志为 `rank-<id>.log`，校验日志为 `verify-<id>.log`；运行或校验失败返回非零退出码。

M 表示每个 rank 的输入和输出行数，输出为 M 行、N 列。卡数必须为 2、4、8、16。脚本中的设备 ID 必须互不重复并按升序排列（当前 CANN 的可见设备列表要求）。所有模式均要求 M/N/K 为正整数；transA 固定为 0，transB 默认为 0。

设 `C` 为当前设备的 AIC 核数，本例要求 `N <= 49152 × max(1, floor(C / M))`：小 M 时按核拆分一行，其余情况整行放入 96 KiB FP16 通信缓冲。二进制会按设备核数检查并打印超限信息。共享数据区要求 `2 × M × N < 1073741824` 字节。

使用 `run.sh` 还要求 `K <= 65535`、`N <= 65535`，这是当前 A2 公共 ACLNN 双标杆调用的内轴限制；同时仍须满足上述内核约束。生成器先按非转置输入计算标杆，再按 transB 转储文件，因此 transB=0/1 都有此限制。

以上容量约束针对每卡固定的 1 GiB SHMEM 数据区，不代表总显存需求；还需为同步标志、A/B/C 和工作空间预留显存，生成双标杆也需要主机内存和 NPU 显存。M/N/K 必须能用 uint32_t 表示，满足容量约束仍需保证实际内存充足。

`run.sh` 用本机回环地址启动所有 rank，适用于单机多卡；它把设备列表写入 `ASCEND_RT_VISIBLE_DEVICES`，并向二进制传入可见设备中的局部编号 `0,1,...`。直接调用二进制时，设备编号也必须与其可见设备范围一致，并为每个 rank 启动一个进程，保持各 rank 的 shape、布局和通信参数一致。

可通过 `CATCCOS_PORT` 设置启动端口（默认 18871），通过 `CATCCOS_TIMEOUT` 设置每个 rank 的运行超时秒数（默认 120，不含数据生成与精度校验）。并发运行多个任务时使用不同端口、输出目录和空闲设备。
