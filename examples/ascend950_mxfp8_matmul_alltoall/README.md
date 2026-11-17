# `ascend950_mxfp8_matmul_alltoall` 示例（切 M 轴 / 切 N 轴）

## 功能说明

- 算子功能：完成 MXFP8 量化的 Matmul 计算与 AlltoAll 通信的融合，**先计算后通信**，
  支持沿 **M 轴**（首维）和沿 **N 轴**（末维）两种 AlltoAll 切分方式：
  - **切 M 轴**：将每卡计算结果沿 M 维均匀分成 `rankSize` 份分发，通信后每卡仍持有
    `(M, N)` 全量结果，但行段来自不同卡（行所有权置换）。
  - **切 N 轴**：将每卡计算结果沿 N 维均匀分成 `rankSize` 份分发，通信后每卡持有
    `(rankSize*M, N/rankSize)`（行聚集 + 列分散），详见 [语义对比](#语义对比)。
- 计算公式：假设每卡 `A_i` 的 shape 为 `(M, K)`（各卡行分片、互不相同），共享权重
  `B` 的 shape 为 `(K, N)`，`rankSize` 为 NPU 卡数，`blockSize = 32`。

  - **切 M 轴**（M-axis split）：

    $$
    computeOut_i = \sum_{0}^{\left \lfloor \frac{K}{blockSize=32} \right \rfloor} (A_i @ B * (aScale_i * bScale)) + bias \\
    permutedOut_i = computeOut_i.view(rankSize,\ M / rankSize,\ N) \\
    output = AlltoAll(permutedOut_i).view(M,\ N)
    $$

    通信语义：
    - 每卡 `i` 计算 `C_i = dequant(A_i) @ dequant(B)` 得到 `(M, N)` 全量结果。
    - 将 `C_i` 沿 M 维均匀分成 `rankSize` 份，`chunkM = M / rankSize`，
      `chunk[j] = C_i[j*chunkM : (j+1)*chunkM, :]` 发给 rank `j`。
    - AlltoAll 后每卡 `j` 持有输出 `D_j`，shape 为 `(M, N)`，其中行块
      `[i*chunkM : (i+1)*chunkM]` 来自 rank `i`（即各卡不同行段拼装而成）。

  - **切 N 轴**（N-axis split）：

    $$
    computeOut_i = \sum_{0}^{\left \lfloor \frac{K}{blockSize=32} \right \rfloor} (A_i @ B * (aScale_i * bScale)) + bias \\
    permutedOut_i = computeOut_i.view(M,\ rankSize,\ N / rankSize).permute(1,\ 0,\ 2) \\
    output = AlltoAll(permutedOut_i).view(rankSize * M,\ N / rankSize)
    $$

    通信语义：
    - 每卡 `i` 计算 `C_i = dequant(A_i) @ dequant(B)` 得到 `(M, N)` 全量结果。
    - 将 `C_i` 沿 N 维均匀分成 `rankSize` 份，`chunkN = N / rankSize`，
      `chunk[j] = C_i[:, j*chunkN : (j+1)*chunkN]` 发给 rank `j`。
    - AlltoAll 后每卡 `j` 持有输出 `D_j`，shape 为 `(rankSize*M, N/rankSize)`，
      其中行块 `[i*M : (i+1)*M]` 来自 rank `i`（列段属于自己的卡）。

## 语义对比

| 维度 | 切 M 轴 | 切 N 轴 |
| :--: | :------------------ | :----------------------------------- |
| 切分轴 | M（首维） | N（末维） |
| reshape | `view(rankSize, M/rankSize, N)` | `view(M, rankSize, N/rankSize).permute(1, 0, 2)` |
| 输出 shape | `(M, N)`（形状不变，行所有权置换） | `(rankSize*M, N/rankSize)`（行聚集 + 列分散） |
| 通信后每卡持有 | 全 M、全 N（行段来自不同卡） | 全 M（拼装）、N/rankSize（列段属于自己的卡） |

## 约束说明

- `rankSize`（NPU 卡数）：支持 2、4、8 卡。
- 切 M 轴时，`M` 必须能被 `rankSize` 整除（均匀 AllToAll 要求），`chunkM = M / rankSize`。
- 切 N 轴时，`N` 必须能被 `rankSize` 整除（均匀 AllToAll 要求），`chunkN = N / rankSize`。
- `K` 建议为 64 的倍数（MXFP8 scale 按 `blockSize=32` 分组对齐要求）。
- 输入 `A_i` 为 `(M, K)`，各卡不同；输入 `B` 为 `(K, N)`，各卡相同（共享权重）。
- 输入/输出数据类型：
  - 输入矩阵 A/B：MX FP8（`float8_e4m3`）
  - 输入 Scale：MX FP8 Scale（`float8_e8m0`），按 `blockSize=32` 分组
  - 输出矩阵 D：`half`（`float16`）

## 使用方式

1. **编译项目**
   进入示例目录并执行编译脚本：
   ```bash
   cd examples/ascend950_mxfp8_matmul_alltoall
   bash scripts/build.sh
   ```

2. **运行 `ascend950_mxfp8_matmul_alltoall` 示例程序**
   在示例目录下执行运行脚本：
   ```bash
   bash scripts/run.sh <device_list>
   ```

   - **参数说明**：
     - `device_list`：指定用于运行的设备（NPU）编号列表，以逗号分隔。
     - 示例：使用第6和第7个NPU设备运行 `ascend950_mxfp8_matmul_alltoall` 双卡示例：
       ```bash
       bash scripts/run.sh 6,7
       ```

   - **切换切分方式**：
      - 切 N 轴（默认）：`scripts/run.sh` 默认调用
        `scripts/gen_data_slice_n.py` 生成 golden 数据，输出 shape 为 `(rankSize*M, N/rankSize)`。
     - 切 M 轴：将 `scripts/run.sh` 中的数据生成脚本由 `gen_data_slice_n.py` 改为
       `scripts/gen_data_slice_m.py`，输出 shape 为 `(M, N)`（行所有权置换）。
       两种方式下 `verify_result.py` 的调用参数保持一致（输出元素总数均为 `M*N`）。

   - **配置计算规模**：
     矩阵形状参数（M、K、N）可在配置文件 `scripts/test_shapes.csv` 中进行设置。
     修改该文件以定义测试用例的输入维度。

   - **数据格式**：
     - 输入矩阵A/B：MX FP8 (float8_e4m3)
     - 输入Scale：MX FP8 Scale (float8_e8m0)
     - 输出矩阵D：half (float16)
