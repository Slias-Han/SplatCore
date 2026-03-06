# SplatCore v0.7 SHA-256 确定性分析文档

## 为什么原版 3DGS 不确定

原版 3DGS 使用 Alpha Blending 的 over-compositing：
    C_final = Σ(α_i · c_i · Π(1 - α_j, j<i))
这要求高斯按深度排序后串行累加。GPU 实现中通常用：
1. Tile-based 并行（每 tile 独立线程块）
2. atomicAdd 做跨线程的 alpha 累加

问题根源：IEEE 754 浮点加法不满足结合律：
    (a + b) + c ≠ a + (b + c)（当 a,b,c 量级差异大时）
线程执行顺序由 GPU 调度器决定，不同帧可能不同，
导致相同输入产生不同的浮点累加顺序 → 不同输出。

## v0.7 的目标：测量，不是修复

v0.7 不要求消除不确定性，要求精确量化：
- 100 帧中，有多少帧 SHA-256 哈希相同？
- 差异发生在哪个 tensor（颜色 vs 深度）？
- 差异的 bit 级分布（1 bit 翻转 vs 多 bit 翻转）？

## GPU 端 SHA-256 实现策略

### 为什么在 GPU 端计算哈希？
CPU 回读（readbackOffscreenFrame）引入 PCIe 传输延迟，
且回读本身是 HOST_VISIBLE 路径，不测量 GPU 显存的真实内容。
GPU 端 SHA-256 compute shader 直接读 VkImage 显存，
与渲染输出之间只有一个 Pipeline Barrier，零传输误差。

### SHA-256 Compute Shader 设计
- input：offscreenImage（RGBA uint32_t，通过 imageLoad 读取）
- output：一个 32 字节（256 bit）的 SSBO（每帧覆写）
- dispatch：每帧渲染完成后立即 dispatch，在 Present 之前
- 实现：标准 SHA-256（FIPS 180-4），纯 GLSL 实现，无扩展依赖

### 哈希序列日志格式
每帧追加写入 sha256_log.txt：
    [Frame 000001] Color SHA256: A3F2...8C01  Depth SHA256: 00FF...1234
    [Frame 000002] Color SHA256: A3F2...8C01  Depth SHA256: 00FF...1234
    [Frame 000003] Color SHA256: B1D4...9E02  ← 哈希变化（不确定性事件）

## 根因分类树

不确定性来源按优先级排查：

P1（最高概率）— atomicAdd 浮点累加顺序
    症状：颜色哈希变化，深度哈希稳定
    定位：fragment shader 中是否有跨调用的 shared/global 浮点累加

P2 — 深度排序不稳定（排序算法非确定性）
    症状：颜色和深度哈希同时变化，且在高斯密集区域集中
    定位：GPU 排序 shader 中是否有不稳定排序（equal-key 的顺序）

P3 — Driver-level 非确定性（调度/缓存）
    症状：低频率随机变化，无空间规律
    定位：Vulkan Validation Layer 的 VK_EXT_pipeline_statistics_query

P4 — Swapchain / Present 时序
    症状：仅在 swapchain 路径存在，offscreen 路径消失
    定位：对比 offscreen SHA256 和 swapchain readback SHA256

## 确定性化方案评估（草案，本版本不实现）

### 方案 A：确定性并行归约树（Deterministic Parallel Reduction）
原理：将浮点累加替换为 balanced binary tree reduction，
确保每次执行的计算树结构完全相同，消除调度顺序影响。
性能代价估计：约 15-30% 吞吐量损失（需要同步点）
M1 可行性：高（已有工业实现，如 Lumen GI）

### 方案 B：定点数累加（Fixed-Point Accumulation）
原理：将 alpha 和颜色转为 Q16.16 或 Q8.24 定点数累加，
整数加法满足结合律，彻底消除浮点非结合问题。
性能代价估计：约 5-10% 损失（整数运算比浮点快或相当）
精度代价：颜色精度从 float32 降到约 16bit，
           对 M1 传感器级输出是否可接受需要实验验证。
M1 可行性：中（精度损失需评估，但汽车行业常用此方案）

### 推荐路径
先做方案 B（快速原型），测量精度损失是否在 M1 传感器误差范围内。
若精度不达标，切方案 A。两方案均不得降低并行度。

## 死亡红线
1. 禁止串行渲染（单线程逐高斯处理）绕过并行不一致
2. SHA-256 必须在 GPU 端计算，不允许 PCIe 回读后 CPU 计算
3. 哈希一致率统计必须基于完整 100 帧，不允许跳帧
4. 根因分析报告必须指向具体 shader 文件和行号，不允许模糊表述
