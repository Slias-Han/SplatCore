// SplatCore MemorySystem.h
// v0.5 内存生命周期宪法
//
// 三级内存区域定义：
// STATIC   — 静态资源区：场景数据，场景切换时释放
// DYNAMIC  — 每帧动态区：排序缓冲区，帧尾强制回收
// STAGING  — 传输暂存区：Staging Buffer，传输完成立即销毁，绝对禁止跨帧复用

#pragma once
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>
#include <cstdint>
#include <string_view>
#include <vector>

namespace SplatCore {

// ─── 内存区域枚举 ──────────────────────────────────────────────────────────
enum class MemoryRegion : uint8_t {
    STATIC  = 0,   // 生命周期：场景级，场景切换时由 SceneManager 统一释放
    DYNAMIC = 1,   // 生命周期：帧级，每帧结束由 FrameAllocator 强制回收
    STAGING = 2,   // 生命周期：传输级，GPU 传输完成 fence 触发后立即释放
};

// ─── VRAM 使用量快照 ───────────────────────────────────────────────────────
// 每帧由 MemorySystem::snapshot() 填充，写入 engine.log
struct VramSnapshot {
    uint64_t staticBytes;   // 静态区当前占用（字节）
    uint64_t dynamicBytes;  // 动态区当前占用（字节）
    uint64_t stagingBytes;  // 暂存区当前占用（字节）
    uint32_t frameIndex;    // 对应帧编号
};

// ─── 分配描述符 ────────────────────────────────────────────────────────────
// 每次 allocate() 调用必须传入此结构体，AllocationName 为必填项
struct AllocationDesc {
    MemoryRegion        region;          // 所属内存区域（必填）
    VkBufferUsageFlags  bufferUsage;     // Buffer 用途标志
    VmaMemoryUsage      vmaUsage;        // VMA 内存使用策略
    VkDeviceSize        size;            // 申请大小（字节）
    std::string_view    allocationName;  // VRAM 追踪标签（必填，禁止传空字符串）
    const VkImageCreateInfo* imageInfo;  // 非空时分配并创建 VkImage
};

// ─── 分配结果 ──────────────────────────────────────────────────────────────
struct Allocation {
    VkBuffer      buffer;
    VkImage       image;
    VmaAllocation vmaAllocation;
    VkDeviceSize  size;
    MemoryRegion  region;
    void*         mappedPtr;  // 若 VMA_ALLOCATION_CREATE_MAPPED_BIT 则非空，否则为 nullptr
    VkFormat      imageFormat = VK_FORMAT_UNDEFINED;
    VkExtent3D    imageExtent{};
    VkImageUsageFlags imageUsage = 0;
};

// ─── MemorySystem 接口声明 ─────────────────────────────────────────────────
class MemorySystem {
public:
    // 初始化：传入 VkInstance / VkPhysicalDevice / VkDevice，创建 VmaAllocator
    // 必须在所有其他 allocate() 调用之前调用，且只能调用一次
    static void init(VkInstance instance,
                     VkPhysicalDevice physicalDevice,
                     VkDevice device);

    // 销毁：释放 VmaAllocator，必须在所有 Allocation 已 free 之后调用
    static void shutdown();

    // 分配接口：根据 desc.region 路由到对应的内存池
    // allocationName 不得为空，违反则触发 std::terminate
    [[nodiscard]]
    static Allocation allocate(const AllocationDesc& desc);

    // 释放接口：仅允许 STATIC 和 STAGING 区域手动调用
    // DYNAMIC 区域的释放只能由 flushDynamicAllocations() 批量执行
    static void free(Allocation& allocation);

    // 帧尾调用：强制回收所有 DYNAMIC 区域分配
    // 渲染循环必须在每帧结束时调用，不得跳过
    static void flushDynamicAllocations();

    // 重置 STATIC 区域生命周期；保留 VkDevice / VmaAllocator。
    static void destroyStaticRegion();
    static void initStaticRegion();

    // 测试辅助：为 STATIC 区域设置临时预算上限。
    // 默认值为无限；仅供内存生命周期测试使用。
    static void setStaticRegionBudgetForTesting(uint64_t budgetBytes);
    static void resetStaticRegionBudgetForTesting();

    // 只读快照：返回当前 STATIC + DYNAMIC 区域内的已分配对象。
    [[nodiscard]]
    static std::vector<Allocation> getAllocations();

    // 快照：填充 VramSnapshot，供日志系统调用
    [[nodiscard]]
    static VramSnapshot snapshot(uint32_t frameIndex);

    // 获取底层 VmaAllocator（仅供需要直接调用 VMA 的系统使用）
    [[nodiscard]]
    static VmaAllocator allocator();

private:
    MemorySystem() = delete;  // 纯静态类，禁止实例化
};

} // namespace SplatCore
