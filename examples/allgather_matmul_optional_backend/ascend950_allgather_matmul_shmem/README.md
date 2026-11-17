# `ascend950_allgather_matmul_shmem` 示例（可选 SHMEM 后端）

该示例使用 `Ascend950AllGatherMatmulWithLocalOptionalBackend` 模板，并选择
`Catccos::Arch::ShmemComm` 作为通信后端。SHMEM 后端负责解析各 rank 的
对称内存地址，通信 block 负责本地 GM 到目标 peer GM 的数据搬运。

## 使用方式

1. **编译项目**
   进入示例目录并执行编译脚本：
   ```bash
   cd examples/allgather_matmul_optional_backend/ascend950_allgather_matmul_shmem
   bash scripts/build.sh
   ```

2. **运行 `ascend950_allgather_matmul_shmem` 示例程序**
   在示例目录下执行运行脚本：
   ```bash
   bash scripts/run.sh <device_list>
   ```

   - **参数说明**：
     - `device_list`：指定用于运行的设备（NPU）编号列表，以逗号分隔。
     - 示例：使用第6和第7个NPU设备运行 `ascend950_allgather_matmul_shmem` 双卡示例：
       ```bash
       bash scripts/run.sh 6,7
       ```

   - **配置计算规模**：
     矩阵形状参数（M、K、N）可在配置文件 `scripts/test_shapes.csv` 中进行设置。
     修改该文件以定义测试用例的输入维度。
