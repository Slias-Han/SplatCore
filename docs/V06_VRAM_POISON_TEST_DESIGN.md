# SplatCore v0.6 VRAM 毒化测试设计文档

## 为什么不用 CUDA？
SplatCore 是纯 Vulkan 引擎，引入 CUDA 会：
1. 增加工具链依赖（需要 CUDA SDK + 驱动版本绑定）
2. 产生 PCIe 传输（CUDA 写完后 Vulkan 读，必须同步）
3. 违反"禁止隐式跨 PCIe 拷贝"原则（M2 前置约束）

**替代方案：Vulkan Compute Shader 毒化**
用 `vkCmdDispatch` 驱动 compute shader，直接写 Vulkan-managed VRAM，
与渲染管线共享同一设备内存空间，零 PCIe 开销。

## 毒化模式定义

| 模式枚举 | 十六进制 | float 解释 | 用途 |
|---------|---------|-----------|------|
| POISON_NAN | 0x7FC00000 | NaN | 触发 NaN 传播检测 |
| POISON_POS_INF | 0x7F800000 | +Inf | 触发 Inf 传播检测 |
| POISON_NEG_INF | 0xFF800000 | -Inf | 触发负 Inf 传播检测 |
| POISON_DEADBEEF | 0xDEADBEEF | 非法浮点 | 触发随机字节检测 |

每次毒化测试按顺序使用全部 4 种模式，各跑一轮。

## 测试流程（两轮对比）

### 干净启动轮（Baseline）
1. 正常初始化引擎
2. 加载测试场景（固定 PLY 文件，SHA-256 锁定）
3. 渲染 100 帧，将最后一帧 Offscreen Buffer 回读到 CPU（`vkCmdCopyImageToBuffer`）
4. 保存为 `baseline_frame.bin`（raw `uint32_t` 数组）

### 毒化启动轮（Poisoned）
1. 启动 Vulkan 设备
2. 在 `MemorySystem::init()` 返回之后、任何业务分配之前，
   用 Compute Shader 毒化所有当前 FREE 内存（通过 VMA 枚举分配）
3. 正常初始化引擎其余部分（与干净轮完全相同代码路径）
4. 加载相同测试场景
5. 渲染 100 帧，回读最后一帧到 CPU
6. 保存为 `poisoned_frame.bin`

### 比对
逐 `uint32_t` 对比 `baseline_frame.bin` 和 `poisoned_frame.bin`。
任何不一致 -> 测试失败，输出 diff 位置和偏差值。

## 关键约束
- 毒化 compute shader 必须在 `MemorySystem::init()` 之后、
  第一个 `vmaCreateAllocator` 分配之前运行（通过 `PoisonTestHarness` 控制）
- 毒化操作必须有 Pipeline Barrier 保证可见性（`srcStageMask = COMPUTE`，
  `dstStageMask = ALL_COMMANDS`，`srcAccessMask = SHADER_WRITE`，
  `dstAccessMask = MEMORY_READ|MEMORY_WRITE`）
- Offscreen Render Target 是新增基础设施（当前引擎只有 Swapchain）
- CI 使用 GitHub Actions + llvmpipe（软渲染），毒化比对结果必须一致

## 死亡红线
1. 毒化 buffer 的生命周期不得超过毒化测试函数作用域
2. baseline 和 poisoned 两轮使用完全相同的 `VkPhysicalDevice` / `VkDevice` 实例
3. 逐帧比对必须是 `uint32_t` 粒度，禁止浮点误差容忍（ε 比对）
4. CI 失败必须 block merge，不允许仅报告
