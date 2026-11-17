### 使用方式

1. **编译项目**
   进入示例目录并执行编译脚本：
   ```bash
   cd examples/allgather_matmul_optional_backend/ascend950_allgather_matmul_hccl
   bash scripts/build.sh
   ```

2. **运行AllGather-MatMul示例程序**
   在示例目录下执行运行脚本：
   ```bash
   bash scripts/run.sh <device_list>
   ```

   - **参数说明**：
     - `device_list`：指定用于运行的设备（NPU）编号列表，以逗号分隔。
     - 示例：使用第6和第7个NPU设备运行2卡AllGather-MatMul示例：
       ```bash
       bash scripts/run.sh 6,7
       ```

   - **配置计算规模**：
     矩阵形状参数（M、K、N）可在配置文件 `scripts/test_shapes.csv` 中进行设置。
     修改该文件以定义测试用例的输入维度。

该示例使用 `Ascend950`/`CATLASS_ARCH=3510` 计算配置。若安装的 CANN
工具包使用Ascend 950PR&950DT系列产品中的具体型号名称，可在编译前覆盖：

```bash
ASCEND950_NPU_MODULE=<msopgen产品名> \
ASCEND950_COMPUTE_UNIT=<CANN计算单元名> \
bash scripts/build.sh
```

通信侧使用 `Arch::Ascend950HcommComm` 后端，通过Ascend 950PR&950DT系列产品支持的 CCU HCCL
高阶 API 分块执行 AllGather，并将结果写入本卡双缓冲 workspace。该实现不依赖
Atlas A2系列产品、Atlas A3系列产品的 `GetWindowsInAddr`、`GetRankDim` 等窗口接口；Host 侧 MC2 原型也显式注册
为 `HcclServerType::CCU`。
