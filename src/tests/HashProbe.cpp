#include "HashProbe.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace SplatCore {

namespace {

struct PushConstants {
    uint32_t imageWidth;
    uint32_t imageHeight;
    uint32_t frameIndex;
};

static_assert(sizeof(PushConstants) == 12,
              "PushConstants must match the shader push_constant layout.");

uint32_t findMemoryTypeIndex(VkPhysicalDevice physicalDevice,
                             uint32_t typeFilter,
                             VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i)
    {
        const bool typeMatches = (typeFilter & (1u << i)) != 0u;
        const bool propertyMatches =
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties;
        if (typeMatches && propertyMatches)
        {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type for HashProbe.");
}

bool hashesMatch(const FrameHash& a, const FrameHash& b)
{
    return std::memcmp(a.colorHash, b.colorHash, sizeof(a.colorHash)) == 0;
}

int countMatchingFramesExcludingBaseline(const std::vector<FrameHash>& hashes,
                                         std::vector<uint32_t>* divergedFrames)
{
    if (divergedFrames != nullptr)
    {
        divergedFrames->clear();
    }
    if (hashes.size() <= 1)
    {
        return 0;
    }

    const FrameHash& baseline = hashes.front();
    int consistentCount = 0;

    for (size_t i = 1; i < hashes.size(); ++i)
    {
        if (hashesMatch(baseline, hashes[i]))
        {
            ++consistentCount;
        }
        else if (divergedFrames != nullptr)
        {
            divergedFrames->push_back(hashes[i].frameIndex);
        }
    }

    return consistentCount;
}

void writeHashWords(std::FILE* file, const uint32_t* words)
{
    for (int i = 0; i < 8; ++i)
    {
        std::fprintf(file, "%08X", words[i]);
        if (i != 7)
        {
            std::fprintf(file, " ");
        }
    }
}

} // namespace

std::vector<uint32_t> HashProbe::loadSpv(const char* path)
{
    std::FILE* file = nullptr;
#ifdef _MSC_VER
    if (fopen_s(&file, path, "rb") != 0)
    {
        file = nullptr;
    }
#else
    file = std::fopen(path, "rb");
#endif
    if (file == nullptr)
    {
        std::terminate();
    }

    if (std::fseek(file, 0, SEEK_END) != 0)
    {
        std::fclose(file);
        std::terminate();
    }

    const long fileSize = std::ftell(file);
    if (fileSize < 0 || (fileSize % static_cast<long>(sizeof(uint32_t))) != 0)
    {
        std::fclose(file);
        std::terminate();
    }

    if (std::fseek(file, 0, SEEK_SET) != 0)
    {
        std::fclose(file);
        std::terminate();
    }

    std::vector<uint32_t> code(static_cast<size_t>(fileSize) / sizeof(uint32_t));
    const size_t readBytes =
        std::fread(code.data(), 1, static_cast<size_t>(fileSize), file);
    std::fclose(file);

    if (readBytes != static_cast<size_t>(fileSize))
    {
        std::terminate();
    }

    return code;
}

void HashProbe::init(VkDevice device,
                     VkPhysicalDevice physicalDevice,
                     VkCommandPool commandPool,
                     VkQueue computeQueue,
                     VkImageView colorImageView,
                     uint32_t imageWidth,
                     uint32_t imageHeight,
                     const char* spvPath)
{
    shutdown();

    m_device = device;
    m_commandPool = commandPool;
    m_computeQueue = computeQueue;
    m_width = imageWidth;
    m_height = imageHeight;
    (void)colorImageView;

    const std::vector<uint32_t> shaderCode = loadSpv(spvPath);

    VkShaderModuleCreateInfo shaderModuleInfo{};
    shaderModuleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleInfo.codeSize = shaderCode.size() * sizeof(uint32_t);
    shaderModuleInfo.pCode = shaderCode.data();

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(m_device, &shaderModuleInfo, nullptr, &shaderModule) !=
        VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create HashProbe shader module.");
    }

    try
    {
        VkDescriptorSetLayoutBinding imageBinding{};
        imageBinding.binding = 0;
        imageBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        imageBinding.descriptorCount = 1;
        imageBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding bufferBinding{};
        bufferBinding.binding = 1;
        bufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bufferBinding.descriptorCount = 1;
        bufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        const VkDescriptorSetLayoutBinding bindings[] = {imageBinding, bufferBinding};

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 2;
        layoutInfo.pBindings = bindings;

        if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr,
                                        &m_descriptorLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create HashProbe descriptor set layout.");
        }

        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_descriptorLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

        if (vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr,
                                   &m_pipelineLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create HashProbe pipeline layout.");
        }

        VkPipelineShaderStageCreateInfo stageInfo{};
        stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = shaderModule;
        stageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = m_pipelineLayout;

        if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                     &m_pipeline) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create HashProbe compute pipeline.");
        }

        VkImageCreateInfo inputImageInfo{};
        inputImageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        inputImageInfo.imageType = VK_IMAGE_TYPE_2D;
        inputImageInfo.extent.width = m_width;
        inputImageInfo.extent.height = m_height;
        inputImageInfo.extent.depth = 1;
        inputImageInfo.mipLevels = 1;
        inputImageInfo.arrayLayers = 1;
        inputImageInfo.format = VK_FORMAT_R8G8B8A8_UINT;
        inputImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        inputImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        inputImageInfo.usage =
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
        inputImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        inputImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateImage(m_device, &inputImageInfo, nullptr, &m_inputImage) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create HashProbe input image.");
        }

        VkMemoryRequirements inputMemRequirements{};
        vkGetImageMemoryRequirements(m_device, m_inputImage, &inputMemRequirements);

        VkMemoryAllocateInfo inputAllocInfo{};
        inputAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        inputAllocInfo.allocationSize = inputMemRequirements.size;
        inputAllocInfo.memoryTypeIndex = findMemoryTypeIndex(
            physicalDevice,
            inputMemRequirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (vkAllocateMemory(m_device, &inputAllocInfo, nullptr, &m_inputMemory) !=
            VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate HashProbe input image memory.");
        }

        if (vkBindImageMemory(m_device, m_inputImage, m_inputMemory, 0) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to bind HashProbe input image memory.");
        }

        VkImageViewCreateInfo inputViewInfo{};
        inputViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        inputViewInfo.image = m_inputImage;
        inputViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        inputViewInfo.format = VK_FORMAT_R8G8B8A8_UINT;
        inputViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        inputViewInfo.subresourceRange.baseMipLevel = 0;
        inputViewInfo.subresourceRange.levelCount = 1;
        inputViewInfo.subresourceRange.baseArrayLayer = 0;
        inputViewInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(m_device, &inputViewInfo, nullptr, &m_inputImageView) !=
            VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create HashProbe input image view.");
        }

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = sizeof(uint32_t) * 8u;
        bufferInfo.usage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(m_device, &bufferInfo, nullptr, &m_hashBuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create HashProbe hash buffer.");
        }

        VkMemoryRequirements memRequirements{};
        vkGetBufferMemoryRequirements(m_device, m_hashBuffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryTypeIndex(
            physicalDevice,
            memRequirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (vkAllocateMemory(m_device, &allocInfo, nullptr, &m_hashMemory) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate HashProbe hash memory.");
        }

        if (vkBindBufferMemory(m_device, m_hashBuffer, m_hashMemory, 0) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to bind HashProbe hash buffer memory.");
        }

        const VkDescriptorPoolSize poolSizes[] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
        };

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes = poolSizes;

        if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool) !=
            VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create HashProbe descriptor pool.");
        }

        VkDescriptorSetAllocateInfo setAllocInfo{};
        setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        setAllocInfo.descriptorPool = m_descriptorPool;
        setAllocInfo.descriptorSetCount = 1;
        setAllocInfo.pSetLayouts = &m_descriptorLayout;

        if (vkAllocateDescriptorSets(m_device, &setAllocInfo, &m_descriptorSet) !=
            VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate HashProbe descriptor set.");
        }

        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageView = m_inputImageView;
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo hashBufferInfo{};
        hashBufferInfo.buffer = m_hashBuffer;
        hashBufferInfo.offset = 0;
        hashBufferInfo.range = sizeof(uint32_t) * 8u;

        VkWriteDescriptorSet descriptorWrites[2]{};
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = m_descriptorSet;
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorWrites[0].pImageInfo = &imageInfo;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = m_descriptorSet;
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[1].pBufferInfo = &hashBufferInfo;

        vkUpdateDescriptorSets(m_device, 2, descriptorWrites, 0, nullptr);

        VkCommandBufferAllocateInfo commandAllocInfo{};
        commandAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandAllocInfo.commandPool = m_commandPool;
        commandAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandAllocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(m_device, &commandAllocInfo, &commandBuffer) !=
            VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate HashProbe init command buffer.");
        }

        try
        {
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to begin HashProbe init command buffer.");
            }

            transitionImageLayout(commandBuffer,
                                  m_inputImage,
                                  VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

            if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to end HashProbe init command buffer.");
            }

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;

            if (vkQueueSubmit(m_computeQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to submit HashProbe init command buffer.");
            }
            if (vkQueueWaitIdle(m_computeQueue) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed waiting for HashProbe init command buffer.");
            }
        }
        catch (...)
        {
            vkFreeCommandBuffers(m_device, m_commandPool, 1, &commandBuffer);
            throw;
        }

        vkFreeCommandBuffers(m_device, m_commandPool, 1, &commandBuffer);
    }
    catch (...)
    {
        vkDestroyShaderModule(m_device, shaderModule, nullptr);
        shutdown();
        throw;
    }

    vkDestroyShaderModule(m_device, shaderModule, nullptr);
}

FrameHash HashProbe::computeHash(VkImage colorImage, uint32_t frameIndex)
{
    if (!isReady())
    {
        throw std::runtime_error("HashProbe is not initialized.");
    }
    if (colorImage == VK_NULL_HANDLE)
    {
        throw std::runtime_error("HashProbe color image is null.");
    }

    VkCommandBufferAllocateInfo commandAllocInfo{};
    commandAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandAllocInfo.commandPool = m_commandPool;
    commandAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandAllocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(m_device, &commandAllocInfo, &commandBuffer) !=
        VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate HashProbe command buffer.");
    }

    try
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to begin HashProbe command buffer.");
        }

        transitionImageLayout(commandBuffer,
                              colorImage,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        transitionImageLayout(commandBuffer,
                              m_inputImage,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        VkImageCopy copyRegion{};
        copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.srcSubresource.mipLevel = 0;
        copyRegion.srcSubresource.baseArrayLayer = 0;
        copyRegion.srcSubresource.layerCount = 1;
        copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.dstSubresource.mipLevel = 0;
        copyRegion.dstSubresource.baseArrayLayer = 0;
        copyRegion.dstSubresource.layerCount = 1;
        copyRegion.extent = {m_width, m_height, 1};

        vkCmdCopyImage(commandBuffer,
                       colorImage,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       m_inputImage,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1,
                       &copyRegion);

        transitionImageLayout(commandBuffer,
                              colorImage,
                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        transitionImageLayout(commandBuffer,
                              m_inputImage,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              VK_IMAGE_LAYOUT_GENERAL);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
        vkCmdBindDescriptorSets(commandBuffer,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                m_pipelineLayout,
                                0, 1, &m_descriptorSet,
                                0, nullptr);

        const PushConstants pushConstants{m_width, m_height, frameIndex};
        vkCmdPushConstants(commandBuffer,
                           m_pipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0,
                           sizeof(PushConstants),
                           &pushConstants);

        vkCmdDispatch(commandBuffer, 1, 1, 1);

        VkMemoryBarrier hashReadyBarrier{};
        hashReadyBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        hashReadyBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        hashReadyBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;

        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT,
                             0,
                             1, &hashReadyBarrier,
                             0, nullptr,
                             0, nullptr);

        transitionImageLayout(commandBuffer,
                              m_inputImage,
                              VK_IMAGE_LAYOUT_GENERAL,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to end HashProbe command buffer.");
        }

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        if (vkQueueSubmit(m_computeQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to submit HashProbe command buffer.");
        }
        if (vkQueueWaitIdle(m_computeQueue) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed waiting for HashProbe command buffer.");
        }
    }
    catch (...)
    {
        vkFreeCommandBuffers(m_device, m_commandPool, 1, &commandBuffer);
        throw;
    }

    vkFreeCommandBuffers(m_device, m_commandPool, 1, &commandBuffer);

    void* mappedData = nullptr;
    if (vkMapMemory(m_device, m_hashMemory, 0, sizeof(uint32_t) * 8u, 0, &mappedData) !=
        VK_SUCCESS)
    {
        throw std::runtime_error("Failed to map HashProbe hash memory.");
    }

    FrameHash frameHash{};
    std::memcpy(frameHash.colorHash, mappedData, sizeof(frameHash.colorHash));
    frameHash.frameIndex = frameIndex;
    vkUnmapMemory(m_device, m_hashMemory);

    return frameHash;
}

float HashProbe::analyzeConsistency(const std::vector<FrameHash>& hashes,
                                    const char* logPath) const
{
    if (hashes.empty())
    {
        return 0.0f;
    }

    std::FILE* file = nullptr;
#ifdef _MSC_VER
    if (fopen_s(&file, logPath, "wb") != 0)
    {
        file = nullptr;
    }
#else
    file = std::fopen(logPath, "wb");
#endif
    if (file == nullptr)
    {
        throw std::runtime_error("Failed to open sha256 log file.");
    }

    const FrameHash& baseline = hashes.front();
    std::fprintf(file, "[Frame %06u] Color SHA256: ", baseline.frameIndex);
    writeHashWords(file, baseline.colorHash);
    std::fprintf(file, "\n");

    int consistentCount = 0;
    for (size_t i = 1; i < hashes.size(); ++i)
    {
        const bool consistent = hashesMatch(baseline, hashes[i]);
        if (consistent)
        {
            ++consistentCount;
        }

        std::fprintf(file, "[Frame %06u] Color SHA256: ", hashes[i].frameIndex);
        writeHashWords(file, hashes[i].colorHash);
        std::fprintf(file,
                     consistent ? " [CONSISTENT]\n"
                                : " [DIVERGED <- \xE4\xB8\x8D\xE4\xB8\x80\xE8\x87\xB4]\n");
    }

    std::fclose(file);

    if (hashes.size() == 1)
    {
        return 1.0f;
    }

    return static_cast<float>(consistentCount) /
           static_cast<float>(hashes.size() - 1u);
}

void HashProbe::generateRootCauseReport(const std::vector<FrameHash>& hashes,
                                        const char* reportPath) const
{
    const int matchingAfterBaseline = countMatchingFramesExcludingBaseline(hashes, nullptr);
    const int totalFrames = static_cast<int>(hashes.size());
    const int consistentFrames =
        totalFrames == 0 ? 0 : matchingAfterBaseline + 1;
    const int divergedFramesCount =
        totalFrames == 0 ? 0 : totalFrames - consistentFrames;
    const float consistencyRate = (totalFrames <= 1)
        ? (totalFrames == 0 ? 0.0f : 1.0f)
        : static_cast<float>(matchingAfterBaseline) /
              static_cast<float>(totalFrames - 1);

    std::vector<uint32_t> divergedFrames;
    countMatchingFramesExcludingBaseline(hashes, &divergedFrames);

    const char* diagnosis = nullptr;
    const char* nextStep1 = nullptr;
    const char* nextStep2 = nullptr;

    if (consistencyRate == 1.0f)
    {
        diagnosis =
            "- 一致率 == 100%：当前渲染路径无可观测非确定性（可能已是确定性路径）";
        nextStep1 =
            "- 核对当前活动图形管线仍指向 shaders/point.vert 和 shaders/point.frag，确认未绕开并行路径。";
        nextStep2 =
            "- 接入真实 3DGS splat shader 后重复 100 帧采样，确认该结果不是由简化渲染路径造成。";
    }
    else if (consistencyRate >= 0.9f)
    {
        diagnosis =
            "- 一致率 90~99%：低频非确定性，疑似 Driver 调度（P3类）";
        nextStep1 =
            "- 对比 offscreen hash 与 swapchain readback，优先排查 main.cpp 中 render-pass 后的 copy/present 路径。";
        nextStep2 =
            "- 启用 VK_EXT_pipeline_statistics_query 并记录同一 workload 的 dispatch/draw 统计，观察 driver-level 抖动。";
    }
    else if (consistencyRate >= 0.5f)
    {
        diagnosis =
            "- 一致率 50~89%：中频非确定性，疑似排序不稳定（P2类）";
        nextStep1 =
            "- 审查 GPU 排序 shader 的 equal-key 处理，确认相同深度 key 不会因并行调度改变顺序。";
        nextStep2 =
            "- 对高斯密集区域单独采样，并将排序输出做逐帧 hash，分离排序阶段与混合阶段的漂移。";
    }
    else
    {
        diagnosis =
            "- 一致率 < 50%：高频非确定性，疑似 atomicAdd 浮点累加（P1类）";
        nextStep1 =
            "- 优先排查 fragment/compute 混合 shader 中的跨线程浮点累加点，确认是否存在 atomicAdd 或 shared reduction。";
        nextStep2 =
            "- 当前仓库活动片段着色器 shaders/point.frag 不含原子操作；若接入 3DGS 路径后出现高频漂移，应先锁定 splat blending shader。";
    }

    std::FILE* file = nullptr;
#ifdef _MSC_VER
    if (fopen_s(&file, reportPath, "wb") != 0)
    {
        file = nullptr;
    }
#else
    file = std::fopen(reportPath, "wb");
#endif
    if (file == nullptr)
    {
        throw std::runtime_error("Failed to open sha256 root-cause report file.");
    }

    std::fprintf(file, "# SHA-256 确定性分析报告\n");
    std::fprintf(file, "## 统计摘要\n");
    std::fprintf(file, "- 测试帧数：%d\n", totalFrames);
    std::fprintf(file, "- 一致帧数：%d\n", consistentFrames);
    std::fprintf(file, "- 不一致帧数：%d\n", divergedFramesCount);
    std::fprintf(file, "- 哈希一致率：%.2f%%\n\n", consistencyRate * 100.0f);

    std::fprintf(file, "## 不一致帧分布\n");
    if (divergedFrames.empty())
    {
        std::fprintf(file, "无\n\n");
    }
    else
    {
        for (uint32_t frame : divergedFrames)
        {
            std::fprintf(file, "- Frame %06u\n", frame);
        }
        std::fprintf(file, "\n");
    }

    std::fprintf(file, "## 初步根因判断\n");
    std::fprintf(file, "%s\n\n", diagnosis);

    std::fprintf(file, "## 下一步行动\n");
    std::fprintf(file, "%s\n", nextStep1);
    std::fprintf(file, "%s\n", nextStep2);

    std::fclose(file);
}

void HashProbe::transitionImageLayout(VkCommandBuffer cmd,
                                      VkImage image,
                                      VkImageLayout oldLayout,
                                      VkImageLayout newLayout)
{
    if (oldLayout == newLayout)
    {
        return;
    }

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags destinationStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL &&
        newLayout == VK_IMAGE_LAYOUT_GENERAL)
    {
        barrier.srcAccessMask =
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL &&
             newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             newLayout == VK_IMAGE_LAYOUT_GENERAL)
    {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_GENERAL &&
             newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask =
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        destinationStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_GENERAL &&
             newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else
    {
        throw std::runtime_error("Unsupported HashProbe image layout transition.");
    }

    vkCmdPipelineBarrier(cmd,
                         sourceStage,
                         destinationStage,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &barrier);
}

void HashProbe::shutdown()
{
    if (m_device == VK_NULL_HANDLE)
    {
        return;
    }

    if (m_hashMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(m_device, m_hashMemory, nullptr);
        m_hashMemory = VK_NULL_HANDLE;
    }
    if (m_hashBuffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(m_device, m_hashBuffer, nullptr);
        m_hashBuffer = VK_NULL_HANDLE;
    }
    if (m_inputImageView != VK_NULL_HANDLE)
    {
        vkDestroyImageView(m_device, m_inputImageView, nullptr);
        m_inputImageView = VK_NULL_HANDLE;
    }
    if (m_inputMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(m_device, m_inputMemory, nullptr);
        m_inputMemory = VK_NULL_HANDLE;
    }
    if (m_inputImage != VK_NULL_HANDLE)
    {
        vkDestroyImage(m_device, m_inputImage, nullptr);
        m_inputImage = VK_NULL_HANDLE;
    }
    if (m_descriptorPool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
    }
    m_descriptorSet = VK_NULL_HANDLE;
    if (m_pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(m_device, m_pipeline, nullptr);
        m_pipeline = VK_NULL_HANDLE;
    }
    if (m_pipelineLayout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
    }
    if (m_descriptorLayout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(m_device, m_descriptorLayout, nullptr);
        m_descriptorLayout = VK_NULL_HANDLE;
    }

    m_height = 0;
    m_width = 0;
    m_commandPool = VK_NULL_HANDLE;
    m_computeQueue = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
}

} // namespace SplatCore
