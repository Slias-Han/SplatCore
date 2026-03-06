#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

namespace SplatCore {

struct FrameHash {
    uint32_t colorHash[8];   // SHA-256 of color tensor (8 x uint32_t)
    uint32_t frameIndex;
};

class HashProbe {
public:
    // Initialize: load sha256_compute.spv and create the compute pipeline.
    // imageWidth/imageHeight: offscreen render target dimensions.
    void init(VkDevice device,
              VkPhysicalDevice physicalDevice,
              VkCommandPool commandPool,
              VkQueue computeQueue,
              VkImageView colorImageView,
              uint32_t imageWidth,
              uint32_t imageHeight,
              const char* spvPath = "shaders/sha256_compute.spv");

    // Call after each rendered frame: dispatch SHA-256, read back the hash,
    // and append it to the caller-owned history.
    // Before calling, colorImage must be transitionable to VK_IMAGE_LAYOUT_GENERAL.
    FrameHash computeHash(VkImage colorImage, uint32_t frameIndex);

    // Write the full hash sequence to a file and return the consistency ratio
    // (0.0~1.0). If the first frame hash matches all subsequent frames,
    // returns 1.0.
    float analyzeConsistency(const std::vector<FrameHash>& hashes,
                             const char* logPath = "sha256_log.txt") const;

    // Generate a root-cause analysis report.
    void generateRootCauseReport(const std::vector<FrameHash>& hashes,
                                 const char* reportPath = "sha256_rootcause.md") const;

    void shutdown();
    bool isReady() const { return m_pipeline != VK_NULL_HANDLE; }

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_computeQueue = VK_NULL_HANDLE;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descriptorLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet m_descriptorSet = VK_NULL_HANDLE;
    VkImage m_inputImage = VK_NULL_HANDLE;
    VkDeviceMemory m_inputMemory = VK_NULL_HANDLE;
    VkImageView m_inputImageView = VK_NULL_HANDLE;
    VkBuffer m_hashBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_hashMemory = VK_NULL_HANDLE;
    uint32_t m_width = 0;
    uint32_t m_height = 0;

    std::vector<uint32_t> loadSpv(const char* path);
    void transitionImageLayout(VkCommandBuffer cmd,
                               VkImage image,
                               VkImageLayout oldLayout,
                               VkImageLayout newLayout);
};

} // namespace SplatCore
