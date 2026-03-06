#include "MemorySystem.h"

#include <atomic>
#include <cstdio>
#include <exception>
#include <mutex>
#include <vector>

namespace SplatCore {

namespace {

VmaAllocator s_allocator = nullptr;
VkDevice s_device = VK_NULL_HANDLE;
std::atomic<uint64_t> s_staticBytes{0};
std::atomic<uint64_t> s_dynamicBytes{0};
std::atomic<uint64_t> s_stagingBytes{0};
std::vector<Allocation> s_dynamicAllocations;
std::mutex s_dynamicMutex;

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

    {
        const std::lock_guard<std::mutex> lock(s_dynamicMutex);
        s_dynamicAllocations.clear();
        s_dynamicAllocations.reserve(256);
    }
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
}

Allocation MemorySystem::allocate(const AllocationDesc& desc)
{
    if (desc.allocationName.empty())
    {
        // AllocationName 为空是架构违规，不允许恢复
        std::terminate();
    }

    if (s_allocator == nullptr)
    {
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
    };

    bytesCounterForRegion(desc.region).fetch_add(allocation.size, std::memory_order_relaxed);

    if (desc.region == MemoryRegion::DYNAMIC)
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
