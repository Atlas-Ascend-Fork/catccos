# AscendTimer 打点工具

基于华为 CANN `GetSystemCycle` 的高精度 NPU 打点计时系统，用于精确测量 AIC/AIV 各阶段执行时间，并自动生成流水线可视化时间线。

## 目录

- [快速开始](#快速开始)
- [编译与运行](#编译与运行)
- [输出文件说明](#输出文件说明)
- [流水线可视化](#流水线可视化)
- [计时项与两种模式](#计时项与两种模式)
- [添加新计时项](#添加新计时项)
- [Kernel 集成指南](#kernel-集成指南)
- [文件清单](#文件清单)

---

## 快速开始

```bash
# 以 matmul_reduce_scatter 样例为例，在仓库根目录执行

# 1. 编译时开启打点
bash examples/matmul_reduce_scatter/scripts/build.sh -DENABLE_TIMER=ON

# 2. 运行程序，参数为逗号分隔的 device_list
#    结束后在样例目录生成 output_timer/
bash examples/matmul_reduce_scatter/scripts/run.sh 0,1

# 3. 后处理生成可视化时间线
python3 tests/dynamic_tiling/utils/process.py examples/matmul_reduce_scatter/output_timer/ mmrs
```

打开 Chrome 浏览器访问 `chrome://tracing`，点击 Load 加载生成的 JSON 文件即可查看流水线排布。

---

## 编译与运行

### 编译选项

| 选项 | 说明 |
|------|------|
| `-DENABLE_TIMER=ON` | 开启打点，添加 `-DENABLE_TIMER --cce-aicore-block-local-init` |

关闭时（默认 OFF），所有打点代码通过 `TIMER_BLOCK` 宏编译为空操作，零开销。

### 运行流程

程序运行后自动执行：

1. `aclrtMalloc` 分配计时缓冲区
2. Kernel 执行期间设备端记录 cycle
3. `aclrtSynchronizeStream` 后，Host 端自动拷贝数据并输出 CSV
4. CSV 文件写入 `output_timer/` 目录

---

## 输出文件说明

运行后在 `output_timer/` 下生成：

```
output_timer/
├── timer_rank_0.csv                        # Duration CSV：每核每项仅 duration(us)
├── timer_rank_1.csv
└── data/
    ├── timer_rank_0_timeline.csv           # Raw CSV：含 start_cycle, end_cycle, duration_us
    └── timer_rank_1_timeline.csv
```

### Duration CSV 格式

```
group_id,core_type,sub_group_id,KERNEL_TIMING_duration(us),AIC_0_duration(us),AIC_1_duration(us),AIV_0_duration(us),AIV_1_duration(us),AIC_IterCount,AIV_IterCount
0,AIC,0,1234.56,100.0,105.0,0,0,2,0
0,AIV,0,1234.56,0,0,95.0,98.0,0,2
0,AIV,1,1234.56,0,0,93.0,96.0,0,2
```

### Raw CSV 格式

比 Duration CSV 多出 start_cycle 和 end_cycle 列，用于后处理生成时间线：

```
group_id,core_type,sub_group_id,KERNEL_TIMING_start_cycle,KERNEL_TIMING_end_cycle,KERNEL_TIMING_duration_us,AIC_0_start_cycle,AIC_0_end_cycle,AIC_0_duration_us,...
```

### 列名规则

| 计时项类型 | 列名格式 | 示例 |
|-----------|---------|------|
| 固定项 | `NAME_duration(us)` | `KERNEL_TIMING_duration(us)` |
| 动态项（Overwrite 模式） | `NAME_iter_duration(us)` | `AIC_0_duration(us)`, `AIV_RS_1_duration(us)` |
| 动态项（Accumulate 模式） | `NAME_duration(us)` | `SYNPIC_TIME_duration(us)`（无迭代后缀） |

---

## 流水线可视化

使用 `process.py` 将 Raw CSV 转为 Chrome Tracing JSON，在 `chrome://tracing` 中可视化流水线排布。

### 用法

```bash
python3 process.py <output_dir> [op_name]
```

| 参数 | 说明 |
|------|------|
| `output_dir` | `output_timer/` 目录路径，默认 `./output_timer/` |
| `op_name` | 算子名称，仅用于生成 JSON 的 `pid` 和输出文件名 |

### 时间戳模式

`process.py` 完全使用 Raw CSV 中的真实 `start_cycle` / `end_cycle` 排布流水线：

排布规则：

- 不再根据算子名称推断 `aic_first` / `aiv_first`
- 不再把同 Step 事件强行对齐到同一 barrier 时间
- 先扫描 Raw CSV 中所有正数 `*_start_cycle`，取最小值作为 0 点
- 事件时间按 `(start_cycle - min_start_cycle) / 50` 换算为 us，保持真实相对时间偏移
- `dur` 优先使用 CSV 中的 `duration_us`，缺失时用 `end_cycle - start_cycle` 换算
- 同一个 `group_id` 的轨道固定放在一起，从上到下为 `AIC`、`AIV_0`、`AIV_1`
- 轨道按 `group_id` 从小到大排列，即 `group 0` 三条轨道后接 `group 1` 三条轨道
- Chrome Tracing 中看到的是采集到的真实时间戳关系，包括天然重叠、空洞和等待
- `AIC`、`AIV` 使用固定颜色；`AIV_RS`、`AIV_AG`、后续新增的 `AIV_*` / `AIC_*` 标签会按标签名前缀分配不同颜色

### 哪些事件进入流水线排布

**所有以 `AIC` 或 `AIV` 开头的动态计时项**都会参与流水线排布。例如：

| 计时项名称 | 是否参与流水线 | 归属 |
|-----------|--------------|------|
| `AIC` | 是 | AIC 轨道 |
| `AIV` | 是 | AIV 轨道 |
| `AIV_RS` | 是 | AIV 轨道（ReduceScatter 阶段） |
| `AIV_AG` | 是 | AIV 轨道（AllGather 阶段） |
| `SYNPIC_TIME` | 否 | 不以 AIC/AIV 开头，仅出现在 CSV 数值列 |

同一轨道上的 `AIV`、`AIV_RS`、`AIV_AG` 等事件按各自真实 `start_cycle` 展示；如果它们在设备侧有重叠或等待，Chrome Tracing 中会直接体现。事件的 `args` 中保留 `group_id`、`core_type`、`sub_group_id`、`start_cycle`、`end_cycle` 和标签前缀，方便回查原始 CSV。

---

## 计时项与两种模式

### 固定项 vs 动态项

| 类型 | 特点 | 用途 |
|------|------|------|
| 固定项 | 整个 Kernel 生命周期只记录一次 | Kernel 总耗时 |
| 动态项 | 每轮循环可多次 Tik/Tok，自动递增迭代号 | AIC 计算、AIV 通信等循环阶段 |

### Overwrite 模式（默认）

每次迭代**独立记录**到不同 slot，CSV 中出现 `TYPE_0`, `TYPE_1`, `TYPE_2`, ...

```cpp
// AIC 阶段：每次循环独立记录
timer.Tik(AscendTimer::AIC);
// ... AIC 工作 ...
timer.Tok<Overwrite>(AscendTimer::AIC);   // 默认模板参数可省略
```

输出：`AIC_0=120us, AIC_1=115us, AIC_2=118us, ...`

### Accumulate 模式

所有迭代的耗时**累加**到同一个 slot，CSV 中只有一列 `NAME`（无迭代后缀）。

```cpp
// 通信等待耗时：累加所有迭代
timer.Tik(AscendTimer::SYNPIC_TIME);
// ... 等待 ...
timer.Tok<Accumulate>(AscendTimer::SYNPIC_TIME);
```

输出：`SYNPIC_TIME=350us`（所有迭代等待时间之和）

**适用场景**：不关心每次迭代的独立耗时，只需总耗时。节省存储空间（仅占 1 个 slot 而非 N 个）。

### 两种模式对比

| | Overwrite | Accumulate |
|---|-----------|------------|
| 存储 | 每次迭代占 1 个 slot | 所有迭代共享 1 个 slot |
| CSV 列名 | `TYPE_0`, `TYPE_1`, ... | `TYPE`（无后缀） |
| 流水线可视化 | 支持（逐迭代排布） | 不参与流水线排布 |
| 适用 | 需要分析每次迭代耗时的阶段 | 只需总耗时的统计项 |

---

## 添加新计时项

### 第一步：在 AscendTimer.hpp 中注册

打开 `tools/AscendTimer.hpp`，在对应列表末尾加一行 `X(YOUR_NAME)`：

**固定项**（整轮一次）：
```cpp
#define ASCEND_TIMER_FIXED_LIST \
    X(KERNEL_TIMING_IDX)    \
    X(YOUR_FIXED_NAME)      \   // 新增
    //继续添加
```

**动态项**（每轮迭代多次）：
```cpp
#define ASCEND_TIMER_DYNAMIC_LIST \
    X(AIC)    \
    X(AIV)    \
    X(AIV_RS) \
    X(AIV_AG) \
    X(YOUR_DYNAMIC_NAME) \   // 新增
    //继续添加
```

添加后，enum 值和 CSV 名称映射自动生成，无需修改其他文件。

### 第二步：在 Kernel 中使用

```cpp
// 固定项
timer.Tik();                                         // 记录起点
// ... 工作代码 ...
timer.Tok<Overwrite>(AscendTimer::YOUR_FIXED_NAME);  // 记录终点

// 动态项（Overwrite 模式）
timer.Tik(AscendTimer::YOUR_DYNAMIC_NAME);
// ... 工作代码 ...
timer.Tok<Overwrite>(AscendTimer::YOUR_DYNAMIC_NAME);

// 动态项（Accumulate 模式）
timer.Tik(AscendTimer::YOUR_DYNAMIC_NAME);
// ... 工作代码 ...
timer.Tok<Accumulate>(AscendTimer::YOUR_DYNAMIC_NAME);
```

### 命名约定

| 规则 | 说明 |
|------|------|
| 仅限字母/数字/下划线 | 不要加逗号或分号 |
| 以 `AIC` 开头 | 该项进入流水线 AIC 轨道 |
| 以 `AIV` 开头 | 该项进入流水线 AIV 轨道 |
| 其他前缀 | 仅出现在 CSV 数值列，不参与流水线排布 |
| 固定项无需前缀 | 固定项不参与流水线，名称自由 |

### 注意事项

- 每个动态项最多支持 **150 次迭代**（`MAX_DYNAMIC_ITER`），超出部分会被丢弃
- 每新增一个动态项，每核存储空间增加 `MAX_DYNAMIC_ITER * 2 * sizeof(int64_t)` = 2400 Bytes
- 核心数由 Ascend C 平台接口 `GetCoreNum()` 获取；该值等价于原来的 `BLOCK_NUM`，timer 内部按 `AIC + 2*AIV` 展开存储行数
- 计时会插入 `PipeBarrier<PIPE_ALL>()`，对流水线有一定影响，正式性能测试请关闭 `ENABLE_TIMER`

---

## Kernel 集成指南

### 标准模式（推荐）

每个 Kernel 类只需一个 `AscendTimerDevice timer` 成员：

```cpp
class MyKernel {
public:
    CATLASS_DEVICE MyKernel() {
    #ifdef ENABLE_TIMER
        __gm__ uint8_t* timer_buffer = GetTimerBuffer();
        if (timer_buffer != nullptr) {
            timer.Init(timer_buffer);
            timer.Tik();                     // Kernel 起始
        }
    #endif
    }

    CATLASS_DEVICE ~MyKernel() {
    #ifdef ENABLE_TIMER
        timer.Tok<Overwrite>(AscendTimer::KERNEL_TIMING_IDX);  // Kernel 结束
    #endif
    }

    // AIC 路径
    CATLASS_DEVICE void runAic(Params &params) {
        for (uint32_t i = 0; i < loops; ++i) {
    #ifdef ENABLE_TIMER
            timer.Tik(AscendTimer::AIC);
    #endif
            // ... AIC 工作代码 ...
    #ifdef ENABLE_TIMER
            timer.Tok<Overwrite>(AscendTimer::AIC);
    #endif
        }
    }

    // AIV 路径
    CATLASS_DEVICE void runAiv(Params &params) {
        for (uint32_t i = 0; i < loops; ++i) {
    #ifdef ENABLE_TIMER
            timer.Tik(AscendTimer::AIV);
    #endif
            // ... AIV 工作代码 ...
    #ifdef ENABLE_TIMER
            timer.Tok<Overwrite>(AscendTimer::AIV);
    #endif
        }
    }

private:
    // ... 其他成员 ...
#ifdef ENABLE_TIMER
    AscendTimerDevice timer;
#endif
};
```

### DeviceDGemm 端（自动）

`DeviceDGemm` 已内置打点支持，无需手动修改：

1. `Run()` 中自动分配 timer buffer 并传给 kernel
2. `aclrtSynchronizeStream` 后自动调用 `ExportTimerCsv()` 输出 CSV
3. 析构时自动 `aclrtFree` timer buffer

---

## 文件清单

| 文件 | 作用 |
|------|------|
| `tools/AscendTimer.hpp` | 公共定义：枚举、常量、宏开关、计时项注册入口 |
| `tools/AscendTimer_device.hpp` | NPU 设备端计时器类 `AscendTimerDevice` |
| `tools/AscendTimer_host.hpp` | Host 端：数据拷贝、CSV 输出 |
| `tests/dynamic_tiling/utils/process.py` | 后处理：CSV → Chrome Tracing JSON 流水线可视化 |
| `include/catccos/dgemm/device/kernel_adapter.hpp` | Kernel 启动适配：设置 timer buffer |
| `include/catccos/dgemm/device/device_dgemm.hpp` | Device 端封装：分配/释放/导出计时数据 |
