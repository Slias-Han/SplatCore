#include "MemorySystem.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <exception>
#include <limits>
#include <mutex>
#include <vector>

namespace SplatCore {

namespace {

VmaAllocator s_allocator = nullptr;
VkDevice s_device = VK_NULL_HANDLE;
std::atomic<uint64_t> s_staticBytes{0};
std::atomic<uint64_t> s_dynamicBytes{0};
std::atomic<uint64_t> s_stagingBytes{0};
std::vector<Allocation> s_staticAllocations;
std::vector<Allocation> s_dynamicAllocations;
std::mutex s_dynamicMutex;
bool s_staticRegionInitialized = false;
std::atomic<uint64_t> s_staticBudgetBytes{std::numeric_limits<uint64_t>::max()};

std::atomic<uint64_t>& bytesCounterForRegion(MemoryRegion region)
{
    switch (region)
    {
    case MemoryRegion::STATIC:
        return s_staticBytes;
    case MemoryRegion::DYNAMIC:
        return s_dynamicBytes;
    case MemoryRegion::STAGING:
        return s_stagingBytes;
    default:
        std::terminate();
    }
}

void eraseTrackedAllocation(std::vector<Allocation>& allocations,
                           VmaAllocation target)
{
    allocations.erase(
        std::remove_if(allocations.begin(),
                       allocations.end(),
                       [target](const Allocation& allocation)
                       {
                           return allocation.vmaAllocation == target;
                       }),
        allocations.end());
}

} // namespace

void MemorySystem::init(VkInstance instance,
                        VkPhysicalDevice physicalDevice,
                        VkDevice device)
{
    if (s_allocator != nullptr)
    {
        std::terminate();
    }

    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_2;
    allocatorInfo.instance = instance;
    allocatorInfo.physicalDevice = physicalDevice;
    allocatorInfo.device = device;

    if (vmaCreateAllocator(&allocatorInfo, &s_allocator) != VK_SUCCESS)
    {
        std::terminate();
    }
    s_device = device;

    s_staticBytes.store(0, std::memory_order_relaxed);
    s_dynamicBytes.store(0, std::memory_order_relaxed);
    s_stagingBytes.store(0, std::memory_order_relaxed);
    s_staticBudgetBytes.store(std::numeric_limits<uint64_t>::max(),
                              std::memory_order_relaxed);

    {
        const std::lock_guard<std::mutex> lock(s_dynamicMutex);
        s_staticAllocations.clear();
        s_staticAllocations.reserve(256);
        s_dynamicAllocations.clear();
        s_dynamicAllocations.reserve(256);
    }
    initStaticRegion();
}

void MemorySystem::shutdown()
{
    flushDynamicAllocations();

    if (s_staticBytes.load(std::memory_order_relaxed) != 0 ||
        s_stagingBytes.load(std::memory_order_relaxed) != 0)
    {
        std::fprintf(stderr,
                     "[MemorySystem] shutdown invariant violated: staticBytes=%llu stagingBytes=%llu\n",
                     static_cast<unsigned long long>(s_staticBytes.load(std::memory_order_relaxed)),
                     static_cast<unsigned long long>(s_stagingBytes.load(std::memory_order_relaxed)));
        std::terminate();
    }

    if (s_allocator != nullptr)
    {
        vmaDestroyAllocator(s_allocator);
        s_allocator = nullptr;
    }
    s_device = VK_NULL_HANDLE;
    s_staticBudgetBytes.store(std::numeric_limits<uint64_t>::max(),
                              std::memory_order_relaxed);
}

Allocation MemorySystem::allocate(const AllocationDesc& desc)
{
    // TODO(P1): replace std::terminate with structured MemoryError return to
    //           allow tests to validate budget overflow without process death
    if (desc.allocationName.empty())
    {
        // AllocationName 为空是架构违规，不允许恢复
        std::terminate();
    }

    if (s_allocator == nullptr)
    {
        std::terminate();
    }
    if (desc.region == MemoryRegion::STATIC && !s_staticRegionInitialized)
    {
        std::fprintf(stderr,
                     "[MemorySystem] allocate rejected: static region is not initialized for %.*s\n",
                     static_cast<int>(desc.allocationName.size()),
                     desc.allocationName.data());
        std::terminate();
    }

    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = desc.vmaUsage;
    allocCreateInfo.flags = VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT;
    allocCreateInfo.pUserData = const_cast<char*>(desc.allocationName.data());
    if (desc.vmaUsage == VMA_MEMORY_USAGE_AUTO_PREFER_HOST ||
        desc.vmaUsage == VMA_MEMORY_USAGE_CPU_ONLY ||
        desc.vmaUsage == VMA_MEMORY_USAGE_CPU_TO_GPU ||
        desc.vmaUsage == VMA_MEMORY_USAGE_GPU_TO_CPU)
    {
        allocCreateInfo.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    }

    VkBuffer buffer = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation vmaAlloc = nullptr;
    VkResult result = VK_ERROR_INITIALIZATION_FAILED;

    if (desc.imageInfo != nullptr)
    {
        if (vkCreateImage(s_device, desc.imageInfo, nullptr, &image) != VK_SUCCESS)
        {
            std::terminate();
        }

        VkMemoryRequirements memoryRequirements{};
        vkGetImageMemoryRequirements(s_device, image, &memoryRequirements);

        result = vmaAllocateMemory(s_allocator,
                                   &memoryRequirements,
                                   &allocCreateInfo,
                                   &vmaAlloc,
                                   nullptr);
        if (result == VK_SUCCESS)
        {
            result = vmaBindImageMemory(s_allocator, vmaAlloc, image);
        }
        if (result != VK_SUCCESS)
        {
            if (vmaAlloc != nullptr)
            {
                vmaFreeMemory(s_allocator, vmaAlloc);
                vmaAlloc = nullptr;
            }
            if (image != VK_NULL_HANDLE)
            {
                vkDestroyImage(s_device, image, nullptr);
                image = VK_NULL_HANDLE;
            }
        }
    }
    else
    {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = desc.size;
        bufferInfo.usage = desc.bufferUsage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(s_device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
        {
            std::terminate();
        }

        VkMemoryRequirements memoryRequirements{};
        vkGetBufferMemoryRequirements(s_device, buffer, &memoryRequirements);

        result = vmaAllocateMemory(s_allocator,
                                   &memoryRequirements,
                                   &allocCreateInfo,
                                   &vmaAlloc,
                                   nullptr);
        if (result == VK_SUCCESS)
        {
            result = vmaBindBufferMemory(s_allocator, vmaAlloc, buffer);
        }
        if (result != VK_SUCCESS)
        {
            if (vmaAlloc != nullptr)
            {
                vmaFreeMemory(s_allocator, vmaAlloc);
                vmaAlloc = nullptr;
            }
            if (buffer != VK_NULL_HANDLE)
            {
                vkDestroyBuffer(s_device, buffer, nullptr);
                buffer = VK_NULL_HANDLE;
            }
        }
    }

    if (result != VK_SUCCESS)
    {
        std::fprintf(stderr,
                     "[MemorySystem] allocate failed: name=%.*s region=%u result=%d\n",
                     static_cast<int>(desc.allocationName.size()),
                     desc.allocationName.data(),
                     static_cast<unsigned>(desc.region),
                     static_cast<int>(result));
        std::terminate();
    }

    VmaAllocationInfo allocInfo{};
    vmaGetAllocationInfo(s_allocator, vmaAlloc, &allocInfo);

    Allocation allocation{
        .buffer = buffer,
        .image = image,
        .vmaAllocation = vmaAlloc,
        .size = desc.imageInfo != nullptr ? allocInfo.size : desc.size,
        .region = desc.region,
        .mappedPtr = allocInfo.pMappedData,
        .imageFormat = desc.imageInfo != nullptr ? desc.imageInfo->format
                                                 : VK_FORMAT_UNDEFINED,
        .imageExtent = desc.imageInfo != nullptr ? desc.imageInfo->extent
                                                 : VkExtent3D{},
        .imageUsage = desc.imageInfo != nullptr ? desc.imageInfo->usage : 0,
    };

    if (desc.region == MemoryRegion::STATIC)
    {
        const uint64_t currentStaticBytes =
            s_staticBytes.load(std::memory_order_relaxed);
        const uint64_t staticBudgetBytes =
            s_staticBudgetBytes.load(std::memory_order_relaxed);
        const bool wouldOverflowBudget =
            allocation.size > (staticBudgetBytes - std::min(currentStaticBytes,
                                                            staticBudgetBytes));

        if (wouldOverflowBudget)
        {
            std::fprintf(stderr,
                         "[MemorySystem] static region budget exceeded: name=%.*s current=%llu request=%llu budget=%llu\n",
                         static_cast<int>(desc.allocationName.size()),
                         desc.allocationName.data(),
                         static_cast<unsigned long long>(currentStaticBytes),
                         static_cast<unsigned long long>(allocation.size),
                         static_cast<unsigned long long>(staticBudgetBytes));

            if (buffer != VK_NULL_HANDLE)
            {
                vkDestroyBuffer(s_device, buffer, nullptr);
            }
            if (image != VK_NULL_HANDLE)
            {
                vkDestroyImage(s_device, image, nullptr);
            }
            if (vmaAlloc != nullptr)
            {
                vmaFreeMemory(s_allocator, vmaAlloc);
            }
            std::terminate();
        }
    }

    bytesCounterForRegion(desc.region).fetch_add(allocation.size, std::memory_order_relaxed);

    if (desc.region == MemoryRegion::STATIC)
    {
        const std::lock_guard<std::mutex> lock(s_dynamicMutex);
        s_staticAllocations.push_back(allocation);
    }
    else if (desc.region == MemoryRegion::DYNAMIC)
    {
        const std::lock_guard<std::mutex> lock(s_dynamicMutex);
        s_dynamicAllocations.push_back(allocation);
    }

    return allocation;
}

void MemorySystem::free(Allocation& allocation)
{
    if (allocation.region == MemoryRegion::DYNAMIC)
    {
        std::terminate();
    }

    if (s_allocator == nullptr)
    {
        std::terminate();
    }

    if (allocation.vmaAllocation != nullptr)
    {
        const VmaAllocation targetAllocation = allocation.vmaAllocation;
        if (allocation.buffer != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(s_device, allocation.buffer, nullptr);
            vmaFreeMemory(s_allocator, allocation.vmaAllocation);
        }
        else if (allocation.image != VK_NULL_HANDLE)
        {
            vkDestroyImage(s_device, allocation.image, nullptr);
            vmaFreeMemory(s_allocator, allocation.vmaAllocation);
        }
        else
        {
            vmaFreeMemory(s_allocator, allocation.vmaAllocation);
        }
        bytesCounterForRegion(allocation.region).fetch_sub(allocation.size, std::memory_order_relaxed);
        if (allocation.region == MemoryRegion::STATIC)
        {
            const std::lock_guard<std::mutex> lock(s_dynamicMutex);
            eraseTrackedAllocation(s_staticAllocations, targetAllocation);
        }
    }

    allocation.buffer = VK_NULL_HANDLE;
    allocation.image = VK_NULL_HANDLE;
    allocation.vmaAllocation = nullptr;
    allocation.mappedPtr = nullptr;
    allocation.size = 0;
}

void MemorySystem::flushDynamicAllocations()
{
    {
        const std::lock_guard<std::mutex> lock(s_dynamicMutex);
        for (Allocation& allocation : s_dynamicAllocations)
        {
            if (allocation.vmaAllocation != nullptr)
            {
                if (s_allocator == nullptr)
                {
                    std::terminate();
                }
                if (allocation.buffer != VK_NULL_HANDLE)
                {
                    vkDestroyBuffer(s_device, allocation.buffer, nullptr);
                    vmaFreeMemory(s_allocator, allocation.vmaAllocation);
                }
                else if (allocation.image != VK_NULL_HANDLE)
                {
                    vkDestroyImage(s_device, allocation.image, nullptr);
                    vmaFreeMemory(s_allocator, allocation.vmaAllocation);
                }
                else
                {
                    vmaFreeMemory(s_allocator, allocation.vmaAllocation);
                }
                allocation.buffer = VK_NULL_HANDLE;
                allocation.image = VK_NULL_HANDLE;
                allocation.vmaAllocation = nullptr;
                allocation.mappedPtr = nullptr;
                allocation.size = 0;
            }
        }
        s_dynamicAllocations.clear();
    }

    s_dynamicBytes.store(0, std::memory_order_relaxed);
}

void MemorySystem::destroyStaticRegion()
{
    if (s_allocator == nullptr)
    {
        std::terminate();
    }

    std::vector<Allocation> allocationsToFree;
    {
        const std::lock_guard<std::mutex> lock(s_dynamicMutex);
        allocationsToFree = s_staticAllocations;
        s_staticAllocations.clear();
        s_staticRegionInitialized = false;
    }

    for (Allocation& allocation : allocationsToFree)
    {
        if (allocation.vmaAllocation == nullptr)
        {
            continue;
        }

        if (allocation.buffer != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(s_device, allocation.buffer, nullptr);
            vmaFreeMemory(s_allocator, allocation.vmaAllocation);
        }
        else if (allocation.image != VK_NULL_HANDLE)
        {
            vkDestroyImage(s_device, allocation.image, nullptr);
            vmaFreeMemory(s_allocator, allocation.vmaAllocation);
        }
        else
        {
            vmaFreeMemory(s_allocator, allocation.vmaAllocation);
        }
    }

    s_staticBytes.store(0, std::memory_order_relaxed);
}

void MemorySystem::initStaticRegion()
{
    if (s_allocator == nullptr)
    {
        std::terminate();
    }

    const std::lock_guard<std::mutex> lock(s_dynamicMutex);
    s_staticAllocations.clear();
    s_staticAllocations.reserve(256);
    s_staticRegionInitialized = true;
}

void MemorySystem::setStaticRegionBudgetForTesting(uint64_t budgetBytes)
{
    s_staticBudgetBytes.store(budgetBytes, std::memory_order_relaxed);
}

void MemorySystem::resetStaticRegionBudgetForTesting()
{
    s_staticBudgetBytes.store(std::numeric_limits<uint64_t>::max(),
                              std::memory_order_relaxed);
}

std::vector<Allocation> MemorySystem::getAllocations()
{
    std::vector<Allocation> allocations;
    const std::lock_guard<std::mutex> lock(s_dynamicMutex);
    allocations.reserve(s_staticAllocations.size() + s_dynamicAllocations.size());
    allocations.insert(allocations.end(),
                       s_staticAllocations.begin(),
                       s_staticAllocations.end());
    allocations.insert(allocations.end(),
                       s_dynamicAllocations.begin(),
                       s_dynamicAllocations.end());
    return allocations;
}

VramSnapshot MemorySystem::snapshot(uint32_t frameIndex)
{
    return VramSnapshot{
        .staticBytes = s_staticBytes.load(std::memory_order_relaxed),
        .dynamicBytes = s_dynamicBytes.load(std::memory_order_relaxed),
        .stagingBytes = s_stagingBytes.load(std::memory_order_relaxed),
        .frameIndex = frameIndex
    };
}

VmaAllocator MemorySystem::allocator()
{
    return s_allocator;
}

} // namespace SplatCore
