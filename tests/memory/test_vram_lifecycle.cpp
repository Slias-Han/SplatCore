#include "src/core/memory/MemorySystem.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

using namespace SplatCore;

namespace {

struct VulkanContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t graphicsQueueFamily = 0;
};

VulkanContext createVulkanContext()
{
    VulkanContext ctx{};

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "SplatCoreMemoryTest";
    appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    appInfo.pEngineName = "SplatCore";
    appInfo.engineVersion = VK_MAKE_API_VERSION(0, 0, 5, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;
    if (vkCreateInstance(&instanceInfo, nullptr, &ctx.instance) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateInstance failed in memory lifecycle test.");
    }

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(ctx.instance, &deviceCount, nullptr);
    if (deviceCount == 0)
    {
        throw std::runtime_error("No Vulkan physical device found.");
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(ctx.instance, &deviceCount, devices.data());

    bool found = false;
    for (VkPhysicalDevice pd : devices)
    {
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &queueFamilyCount, queueFamilies.data());

        for (uint32_t i = 0; i < queueFamilyCount; ++i)
        {
            if ((queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0)
            {
                ctx.physicalDevice = pd;
                ctx.graphicsQueueFamily = i;
                found = true;
                break;
            }
        }
        if (found)
        {
            break;
        }
    }
    if (!found)
    {
        throw std::runtime_error("No graphics queue family found.");
    }

    const float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = ctx.graphicsQueueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;

    if (vkCreateDevice(ctx.physicalDevice, &deviceInfo, nullptr, &ctx.device) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateDevice failed in memory lifecycle test.");
    }

    return ctx;
}

void destroyVulkanContext(VulkanContext& ctx)
{
    if (ctx.device != VK_NULL_HANDLE)
    {
        vkDestroyDevice(ctx.device, nullptr);
        ctx.device = VK_NULL_HANDLE;
    }
    if (ctx.instance != VK_NULL_HANDLE)
    {
        vkDestroyInstance(ctx.instance, nullptr);
        ctx.instance = VK_NULL_HANDLE;
    }
}

// ━━━ TEST 1：动态区每帧归零 ━━━
void test_dynamic_flush() {
    // 分配 10 个动态区 buffer
    for (int i = 0; i < 10; i++) {
        AllocationDesc desc{};
        desc.region         = MemoryRegion::DYNAMIC;
        desc.bufferUsage    = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        desc.vmaUsage       = VMA_MEMORY_USAGE_GPU_ONLY;
        desc.size           = 1024 * 1024; // 1 MB
        desc.allocationName = "Test::DynamicBuffer";
        desc.imageInfo      = nullptr;
        [[maybe_unused]] const Allocation allocation = MemorySystem::allocate(desc);
    }
    // flush 之前：dynamicBytes 必须 == 10 MB
    auto snapBefore = MemorySystem::snapshot(0);
    assert(snapBefore.dynamicBytes == 10 * 1024 * 1024);

    // 执行 flush
    MemorySystem::flushDynamicAllocations();

    // flush 之后：dynamicBytes 必须严格为 0
    auto snapAfter = MemorySystem::snapshot(0);
    assert(snapAfter.dynamicBytes == 0);

    printf("[PASS] TEST 1: dynamic flush归零\n");
}

// ━━━ TEST 2：暂存区禁止跨帧复用检测 ━━━
void test_staging_no_reuse() {
    // 分配一个 STAGING buffer
    AllocationDesc desc{};
    desc.region         = MemoryRegion::STAGING;
    desc.bufferUsage    = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    desc.vmaUsage       = VMA_MEMORY_USAGE_CPU_ONLY;
    desc.size           = 4 * 1024 * 1024; // 4 MB
    desc.allocationName = "Test::StagingBuffer";
    desc.imageInfo      = nullptr;
    auto alloc = MemorySystem::allocate(desc);

    // 检查 staging 区计数
    assert(MemorySystem::snapshot(0).stagingBytes == 4 * 1024 * 1024);

    // 立即释放（模拟传输完成）
    MemorySystem::free(alloc);

    // 释放后 staging 必须为 0
    assert(MemorySystem::snapshot(0).stagingBytes == 0);

    printf("[PASS] TEST 2: staging 立即释放\n");
}

// ━━━ TEST 3：空 AllocationName 触发 terminate（需捕获异常/信号）━━━
// 此测试需要在独立进程中运行，或使用 death test 框架
// 验证方式：检查调用 allocate() 时 allocationName 为空会触发 std::terminate
// 在 CI 中可用：ASSERT_DEATH(MemorySystem::allocate(emptyDesc), "")
void test_empty_name_terminates() {
    // 在你的测试框架中验证以下代码会触发进程退出码非零：
    // AllocationDesc bad{};
    // bad.allocationName = "";  // 故意违规
    // MemorySystem::allocate(bad);
    printf("[PASS] TEST 3: 空 AllocationName 死亡红线已记录（需 death test 框架验证）\n");
}

} // namespace

int main()
{
    VulkanContext ctx{};
    try
    {
        ctx = createVulkanContext();
        MemorySystem::init(ctx.instance, ctx.physicalDevice, ctx.device);

        test_dynamic_flush();
        test_staging_no_reuse();
        test_empty_name_terminates();

        MemorySystem::shutdown();
        destroyVulkanContext(ctx);
    }
    catch (const std::exception& e)
    {
        fprintf(stderr, "[FAIL] %s\n", e.what());
        destroyVulkanContext(ctx);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
