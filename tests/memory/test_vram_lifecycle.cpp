#include "src/core/memory/MemorySystem.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace SplatCore;

namespace {

constexpr uint64_t kDeathTestStaticBudgetBytes = 1024 * 1024;
constexpr uint64_t kDeathTestRequestBytes = kDeathTestStaticBudgetBytes + 1024;

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

int run_over_budget_child()
{
    VulkanContext ctx = createVulkanContext();
    MemorySystem::init(ctx.instance, ctx.physicalDevice, ctx.device);
    MemorySystem::setStaticRegionBudgetForTesting(kDeathTestStaticBudgetBytes);

    AllocationDesc desc{};
    desc.region = MemoryRegion::STATIC;
    desc.bufferUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    desc.vmaUsage = VMA_MEMORY_USAGE_GPU_ONLY;
    desc.size = kDeathTestRequestBytes;
    desc.allocationName = "Test::StaticOverBudget";
    desc.imageInfo = nullptr;

    std::printf("[TEST 3 CHILD] Triggering over-budget static allocation (%llu > %llu)\n",
                static_cast<unsigned long long>(desc.size),
                static_cast<unsigned long long>(kDeathTestStaticBudgetBytes));
    std::fflush(stdout);

    [[maybe_unused]] const Allocation allocation = MemorySystem::allocate(desc);

    MemorySystem::resetStaticRegionBudgetForTesting();
    MemorySystem::shutdown();
    destroyVulkanContext(ctx);
    return EXIT_SUCCESS;
}

bool test_static_budget_death(const char* executablePath)
{
    if (executablePath == nullptr || executablePath[0] == '\0')
    {
        std::printf("[FAIL] TEST 3: executable path unavailable\n");
        return false;
    }

#ifdef _WIN32
    std::string commandLine = "\"";
    commandLine += executablePath;
    commandLine += "\" --death-test-overbudget";

    STARTUPINFOA startupInfo{};
    startupInfo.cb = sizeof(startupInfo);
    PROCESS_INFORMATION processInfo{};
    std::vector<char> mutableCommandLine(commandLine.begin(), commandLine.end());
    mutableCommandLine.push_back('\0');

    const BOOL created = CreateProcessA(
        nullptr,
        mutableCommandLine.data(),
        nullptr,
        nullptr,
        FALSE,
        0,
        nullptr,
        nullptr,
        &startupInfo,
        &processInfo);

    if (!created)
    {
        std::printf("[FAIL] TEST 3: CreateProcess failed (%lu)\n",
                    static_cast<unsigned long>(GetLastError()));
        return false;
    }

    WaitForSingleObject(processInfo.hProcess, INFINITE);

    DWORD exitCode = 0;
    const BOOL gotExitCode = GetExitCodeProcess(processInfo.hProcess, &exitCode);
    CloseHandle(processInfo.hThread);
    CloseHandle(processInfo.hProcess);

    if (!gotExitCode)
    {
        std::printf("[FAIL] TEST 3: GetExitCodeProcess failed (%lu)\n",
                    static_cast<unsigned long>(GetLastError()));
        return false;
    }

    if (exitCode != EXIT_SUCCESS)
    {
        std::printf("[PASS] TEST 3 PASS: over-budget allocation terminated child process (exit=%lu)\n",
                    static_cast<unsigned long>(exitCode));
        return true;
    }

    std::printf("[FAIL] TEST 3: child survived over-budget allocation\n");
    return false;
#else
    const pid_t pid = fork();
    if (pid < 0)
    {
        std::printf("[FAIL] TEST 3: fork failed\n");
        return false;
    }
    if (pid == 0)
    {
        execl(executablePath, executablePath, "--death-test-overbudget", nullptr);
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0)
    {
        std::printf("[FAIL] TEST 3: waitpid failed\n");
        return false;
    }

    if (WIFSIGNALED(status) || (WIFEXITED(status) && WEXITSTATUS(status) != EXIT_SUCCESS))
    {
        std::printf("[PASS] TEST 3 PASS: over-budget allocation terminated child process\n");
        return true;
    }

    std::printf("[FAIL] TEST 3: child survived over-budget allocation\n");
    return false;
#endif
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc >= 2 && std::string(argv[1]) == "--death-test-overbudget")
    {
        return run_over_budget_child();
    }

    VulkanContext ctx{};
    try
    {
        ctx = createVulkanContext();
        MemorySystem::init(ctx.instance, ctx.physicalDevice, ctx.device);

        test_dynamic_flush();
        test_staging_no_reuse();
        if (!test_static_budget_death(argv[0]))
        {
            MemorySystem::resetStaticRegionBudgetForTesting();
            MemorySystem::shutdown();
            destroyVulkanContext(ctx);
            return EXIT_FAILURE;
        }

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
