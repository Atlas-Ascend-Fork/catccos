# Ascend950 MXFP8 AllToAll MatMul Split-K URMA

本示例演示 Ascend950 上基于 URMA 通信的 MXFP8 split-K AllToAll MatMul，并提供默认实现和
原子累加实现。

## 功能说明

每个 rank 持有 A 的一个 K 维分片。经过 AllToAll 通信和矩阵乘计算后，各 rank 分别输出 C 的一个
M 维分片：

```text
输入 A:      [M, localK]
逻辑输入 B:  [localK * rankSize, N]
输出 C:      [M / rankSize, N]
```

B 在 GM 中按 `[N, localK * rankSize]` 存储，kernel 按逻辑形状
`[localK * rankSize, N]` 读取。A scale 和 B scale 使用 MX packed scale，K 方向每 32 个元素共享一个
scale。

## 实现选项

| 实现 | 构建及运行选项 | 说明 |
| --- | --- | --- |
| 默认实现 | `default` | 对各 K 分片的计算结果进行累加后输出 |
| 原子累加实现 | `local_atomic` | 本地分片结果直接写入输出，其他 rank 的分片结果通过 FP16 atomic add 累加 |

原子累加实现的 FP16 舍入路径与默认实现不同。运行脚本会根据所选实现使用相应的精度基准进行校验。

## 构建与运行

在示例目录下构建并运行默认实现：

```bash
cd examples/ascend950_mxfp8_alltoall_matmul_split_k_urma
bash scripts/build.sh
bash scripts/run.sh 0,1,2,3
```

构建并运行原子累加实现：

```bash
bash scripts/build.sh local_atomic
CATCCOS_SPLIT_K_URMA_VARIANT=local_atomic bash scripts/run.sh 0,1,2,3
```

如需同时构建两种实现，可执行：

```bash
bash scripts/build.sh all
```

运行脚本的参数为逗号分隔的设备 ID 列表，例如：

```bash
bash scripts/run.sh 12,13,14,15
```

## 精度校验

`scripts/run.sh` 会为每组测试 shape 生成输入和参考结果，执行多 rank 算子，并自动完成精度校验。
任意一组校验失败时，脚本将返回非零状态。

## Shape 配置与约束

测试 shape 在 `scripts/test_shapes.csv` 中配置，字段为 `M,localK,N`。当前示例要求：

- `M` 能被 `rankSize` 整除；
- `localK` 能被 512 整除；
- `rankSize` 不大于 8。
