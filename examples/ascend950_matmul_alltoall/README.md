### 使用方式

1. **编译项目**
   进入示例目录并执行编译脚本：
   ```bash
   cd examples/ascend950_matmul_alltoall
   bash scripts/build.sh
   ```

2. **运行 `ascend950_matmul_alltoall` 示例程序**
   在示例目录下执行运行脚本：
   ```bash
   bash scripts/run.sh <device_list>
   ```

   - **参数说明**：
     - `device_list`：指定用于运行的设备（NPU）编号列表，以逗号分隔。
     - 示例：使用第6和第7个NPU设备运行2卡示例：
       ```bash
       bash scripts/run.sh 6,7
       ```

   - **配置计算规模**：
     矩阵形状参数（M、K、N）可在配置文件 `scripts/test_shapes.csv` 中进行设置。

   - **注意事项**：
     - M 必须能被 rankSize 整除（均匀 AllToAll 要求）。
     - 先做 MatMul: C = A (M×K) × B (K×N)，得到 M×N 结果。
     - 再做 AllToAll: 将 C 沿 M 维度均匀分成 R 份，chunk[j] 发给 rank j。
     - 最终每个 rank 的输出 D 为 M×N，由来自所有 rank 的不同行段拼装而成。
