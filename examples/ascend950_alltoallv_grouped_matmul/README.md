### 使用方式

1. **编译项目**
   进入示例目录并执行编译脚本：
   ```bash
   cd examples/ascend950_alltoallv_grouped_matmul
   bash scripts/build.sh
   ```

2. **运行 `ascend950_alltoallv_grouped_matmul` 示例程序**
   在示例目录下执行运行脚本：
   ```bash
   bash scripts/run.sh <device_list> [ep_size] [expert_num]
   ```

   - **参数说明**：
     - `device_list`: 指定用于运行的设备（NPU）编号列表，以逗号分隔。

     - `ep_size`: 指定专家并行数，需与用于运行的设备(NPU)数量保持一致，未显示指定时默认值为2。

     - `expert_num`: 指定专家数，未显示指定时默认值为8。

        示例：使用第6和第7个NPU设备运行 `ascend950_alltoallv_grouped_matmul` 双卡示例，专家并行数为2，专家数为8：
       ```bash
       # 修改运行参数配置时可能需要同步shape文件
       bash scripts/run.sh 6,7 2 4
       ```

   - **配置计算规模**：
     矩阵形状参数（M、K、N）可在配置文件 `scripts/test_shapes.csv` 中进行设置。
     修改该文件以定义测试用例的输入维度。
