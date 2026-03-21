#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

namespace SplatCore {

struct Allocation;

enum class PoisonPattern : uint32_t {
    NAN_PATTERN = 0x7FC00000u,
    POS_INF = 0x7F800000u,
    NEG_INF = 0xFF800000u,
    DEAD_BEEF = 0xDEADBEEFu,
};

class PoisonTestHarness {
public:
    void init(VkDevice device,
              VkPhysicalDevice physicalDevice,
              VkCommandPool commandPool,
              VkQueue computeQueue,
              const char* spvPath = "shaders/vram_poison_comp.spv");

    void poisonBuffer(VkBuffer targetBuffer,
                      VkDeviceSize bufferSize,
                      PoisonPattern pattern);

    void poisonAll(PoisonPattern pattern);

    void shutdown();

    bool isReady() const { return m_pipeline != VK_NULL_HANDLE; }

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_computeQueue = VK_NULL_HANDLE;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;

    std::vector<uint32_t> loadSpv(const char* path);
    void insertMemoryBarrier(VkCommandBuffer cmd);
    void fillHostVisibleAllocation(const Allocation& allocation,
                                   PoisonPattern pattern);
    void poisonImage(const Allocation& allocation,
                     PoisonPattern pattern);
};

} // namespace SplatCore
