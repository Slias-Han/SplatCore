#include "HashProbe.h"
#include "PcieTransferTracker.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

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

VkShaderModule createShaderModule(VkDevice device, const std::vector<uint32_t>& code)
{
    VkShaderModuleCreateInfo shaderModuleInfo{};
    shaderModuleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleInfo.codeSize = code.size() * sizeof(uint32_t);
    shaderModuleInfo.pCode = code.data();

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &shaderModuleInfo, nullptr, &shaderModule) !=
        VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create HashProbe shader module.");
    }

    return shaderModule;
}

void createHostVisibleHashBuffer(VkDevice device,
                                 VkPhysicalDevice physicalDevice,
                                 VkBuffer& buffer,
                                 VkDeviceMemory& memory)
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = sizeof(uint32_t) * 8u;
    bufferInfo.usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create HashProbe hash buffer.");
    }

    VkMemoryRequirements memRequirements{};
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryTypeIndex(
        physicalDevice,
        memRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate HashProbe hash memory.");
    }

    if (vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to bind HashProbe hash buffer memory.");
    }
}

std::string deriveSiblingSpvPath(const char* basePath, const char* siblingFileName)
{
    if (basePath == nullptr || basePath[0] == '\0')
    {
        return siblingFileName;
    }

    std::string path = basePath;
    const size_t slashPos = path.find_last_of("/\\");
    if (slashPos == std::string::npos)
    {
        return siblingFileName;
    }

    path.resize(slashPos + 1u);
    path += siblingFileName;
    return path;
}

bool colorHashesMatch(const FrameHash& a, const FrameHash& b)
{
    return std::memcmp(a.colorHash, b.colorHash, sizeof(a.colorHash)) == 0;
}

bool depthHashesMatch(const FrameHash& a, const FrameHash& b)
{
    return a.depthHash == b.depthHash;
}

bool hasDepthHashes(const std::vector<FrameHash>& hashes)
{
    for (const FrameHash& hash : hashes)
    {
        if (hash.depthHash != 0u)
        {
            return true;
        }
    }
    return false;
}

using HashMatchFn = bool (*)(const FrameHash&, const FrameHash&);

int countMatchingFramesExcludingBaseline(const std::vector<FrameHash>& hashes,
                                         HashMatchFn matches,
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
        if (matches(baseline, hashes[i]))
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

int countMatchingFramesIncludingBaseline(const std::vector<FrameHash>& hashes,
                                         HashMatchFn matches)
{
    if (hashes.empty())
    {
        return 0;
    }

    return countMatchingFramesExcludingBaseline(hashes, matches, nullptr) + 1;
}

float computeConsistencyExcludingBaseline(const std::vector<FrameHash>& hashes,
                                          HashMatchFn matches)
{
    if (hashes.empty())
    {
        return 0.0f;
    }
    if (hashes.size() == 1u)
    {
        return 1.0f;
    }

    return static_cast<float>(
               countMatchingFramesExcludingBaseline(hashes, matches, nullptr)) /
           static_cast<float>(hashes.size() - 1u);
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

const char* combinedStatusLabel(bool colorMatch, bool depthMeasured, bool depthMatch)
{
    if (!depthMeasured)
    {
        return colorMatch ? "[CONSISTENT]" : "[DIVERGED <- 不一致]";
    }
    if (colorMatch && depthMatch)
    {
        return "[CONSISTENT]";
    }
    if (!colorMatch && !depthMatch)
    {
        return "[DIVERGED <- COLOR, DEPTH]";
    }
    return colorMatch ? "[DIVERGED <- DEPTH]" : "[DIVERGED <- COLOR]";
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
                     const char* spvPath,
                     VkImageView depthImageView)
{
    shutdown();

    m_device = device;
    m_commandPool = commandPool;
    m_computeQueue = computeQueue;
    m_width = imageWidth;
    m_height = imageHeight;
    m_depthImageView = depthImageView;
    (void)colorImageView;

    const std::vector<uint32_t> shaderCode = loadSpv(spvPath);
    VkShaderModule colorShaderModule = createShaderModule(m_device, shaderCode);

    VkShaderModule depthShaderModule = VK_NULL_HANDLE;
    try
    {
        if (m_depthImageView != VK_NULL_HANDLE)
        {
            const std::string depthSpvPath =
                deriveSiblingSpvPath(spvPath, "sha256_depth_compute.spv");
            const std::vector<uint32_t> depthShaderCode = loadSpv(depthSpvPath.c_str());
            depthShaderModule = createShaderModule(m_device, depthShaderCode);
        }

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
        stageInfo.module = colorShaderModule;
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

        createHostVisibleHashBuffer(m_device,
                                    physicalDevice,
                                    m_hashBuffer,
                                    m_hashMemory);

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

        if (depthShaderModule != VK_NULL_HANDLE)
        {
            VkSamplerCreateInfo samplerInfo{};
            samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            samplerInfo.magFilter = VK_FILTER_NEAREST;
            samplerInfo.minFilter = VK_FILTER_NEAREST;
            samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.maxAnisotropy = 1.0f;
            samplerInfo.minLod = 0.0f;
            samplerInfo.maxLod = 0.0f;
            samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
            samplerInfo.compareEnable = VK_FALSE;

            if (vkCreateSampler(m_device, &samplerInfo, nullptr, &m_depthSampler) !=
                VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create HashProbe depth sampler.");
            }

            VkDescriptorSetLayoutBinding depthImageBinding{};
            depthImageBinding.binding = 0;
            depthImageBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            depthImageBinding.descriptorCount = 1;
            depthImageBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            VkDescriptorSetLayoutBinding depthBufferBinding{};
            depthBufferBinding.binding = 1;
            depthBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            depthBufferBinding.descriptorCount = 1;
            depthBufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            const VkDescriptorSetLayoutBinding depthBindings[] = {
                depthImageBinding,
                depthBufferBinding};

            VkDescriptorSetLayoutCreateInfo depthLayoutInfo{};
            depthLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            depthLayoutInfo.bindingCount = 2;
            depthLayoutInfo.pBindings = depthBindings;

            if (vkCreateDescriptorSetLayout(m_device,
                                            &depthLayoutInfo,
                                            nullptr,
                                            &m_depthDescriptorLayout) != VK_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to create HashProbe depth descriptor set layout.");
            }

            VkPipelineLayoutCreateInfo depthPipelineLayoutInfo{};
            depthPipelineLayoutInfo.sType =
                VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            depthPipelineLayoutInfo.setLayoutCount = 1;
            depthPipelineLayoutInfo.pSetLayouts = &m_depthDescriptorLayout;
            depthPipelineLayoutInfo.pushConstantRangeCount = 1;
            depthPipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

            if (vkCreatePipelineLayout(m_device,
                                       &depthPipelineLayoutInfo,
                                       nullptr,
                                       &m_depthPipelineLayout) != VK_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to create HashProbe depth pipeline layout.");
            }

            VkPipelineShaderStageCreateInfo depthStageInfo{};
            depthStageInfo.sType =
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            depthStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            depthStageInfo.module = depthShaderModule;
            depthStageInfo.pName = "main";

            VkComputePipelineCreateInfo depthPipelineInfo{};
            depthPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            depthPipelineInfo.stage = depthStageInfo;
            depthPipelineInfo.layout = m_depthPipelineLayout;

            if (vkCreateComputePipelines(m_device,
                                         VK_NULL_HANDLE,
                                         1,
                                         &depthPipelineInfo,
                                         nullptr,
                                         &m_depthPipeline) != VK_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to create HashProbe depth compute pipeline.");
            }

            createHostVisibleHashBuffer(m_device,
                                        physicalDevice,
                                        m_depthHashBuffer,
                                        m_depthHashMemory);

            const VkDescriptorPoolSize depthPoolSizes[] = {
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
            };

            VkDescriptorPoolCreateInfo depthPoolInfo{};
            depthPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            depthPoolInfo.maxSets = 1;
            depthPoolInfo.poolSizeCount = 2;
            depthPoolInfo.pPoolSizes = depthPoolSizes;

            if (vkCreateDescriptorPool(m_device,
                                       &depthPoolInfo,
                                       nullptr,
                                       &m_depthDescriptorPool) != VK_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to create HashProbe depth descriptor pool.");
            }

            VkDescriptorSetAllocateInfo depthSetAllocInfo{};
            depthSetAllocInfo.sType =
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            depthSetAllocInfo.descriptorPool = m_depthDescriptorPool;
            depthSetAllocInfo.descriptorSetCount = 1;
            depthSetAllocInfo.pSetLayouts = &m_depthDescriptorLayout;

            if (vkAllocateDescriptorSets(m_device,
                                         &depthSetAllocInfo,
                                         &m_depthDescriptorSet) != VK_SUCCESS)
            {
                throw std::runtime_error(
                    "Failed to allocate HashProbe depth descriptor set.");
            }

            VkDescriptorImageInfo depthImageInfo{};
            depthImageInfo.sampler = m_depthSampler;
            depthImageInfo.imageView = m_depthImageView;
            depthImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorBufferInfo depthHashBufferInfo{};
            depthHashBufferInfo.buffer = m_depthHashBuffer;
            depthHashBufferInfo.offset = 0;
            depthHashBufferInfo.range = sizeof(uint32_t) * 8u;

            VkWriteDescriptorSet depthDescriptorWrites[2]{};
            depthDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            depthDescriptorWrites[0].dstSet = m_depthDescriptorSet;
            depthDescriptorWrites[0].dstBinding = 0;
            depthDescriptorWrites[0].descriptorCount = 1;
            depthDescriptorWrites[0].descriptorType =
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            depthDescriptorWrites[0].pImageInfo = &depthImageInfo;

            depthDescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            depthDescriptorWrites[1].dstSet = m_depthDescriptorSet;
            depthDescriptorWrites[1].dstBinding = 1;
            depthDescriptorWrites[1].descriptorCount = 1;
            depthDescriptorWrites[1].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            depthDescriptorWrites[1].pBufferInfo = &depthHashBufferInfo;

            vkUpdateDescriptorSets(m_device, 2, depthDescriptorWrites, 0, nullptr);
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

            if (vkQueueSubmit(m_computeQueue, 1, &submitInfo, VK_NULL_HANDLE) !=
                VK_SUCCESS)
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
        if (depthShaderModule != VK_NULL_HANDLE)
        {
            vkDestroyShaderModule(m_device, depthShaderModule, nullptr);
        }
        vkDestroyShaderModule(m_device, colorShaderModule, nullptr);
        shutdown();
        throw;
    }

    if (depthShaderModule != VK_NULL_HANDLE)
    {
        vkDestroyShaderModule(m_device, depthShaderModule, nullptr);
    }
    vkDestroyShaderModule(m_device, colorShaderModule, nullptr);
}

FrameHash HashProbe::computeHash(VkImage colorImage,
                                 uint32_t frameIndex,
                                 VkImage depthImage)
{
    if (!isReady())
    {
        throw std::runtime_error("HashProbe is not initialized.");
    }
    if (colorImage == VK_NULL_HANDLE)
    {
        throw std::runtime_error("HashProbe color image is null.");
    }

    const bool hashDepth =
        depthImage != VK_NULL_HANDLE && m_depthPipeline != VK_NULL_HANDLE &&
        m_depthDescriptorSet != VK_NULL_HANDLE;

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

        transitionImageLayout(commandBuffer,
                              m_inputImage,
                              VK_IMAGE_LAYOUT_GENERAL,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        if (hashDepth)
        {
            transitionImageLayout(commandBuffer,
                                  depthImage,
                                  VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_IMAGE_ASPECT_DEPTH_BIT);

            vkCmdBindPipeline(commandBuffer,
                              VK_PIPELINE_BIND_POINT_COMPUTE,
                              m_depthPipeline);
            vkCmdBindDescriptorSets(commandBuffer,
                                    VK_PIPELINE_BIND_POINT_COMPUTE,
                                    m_depthPipelineLayout,
                                    0, 1, &m_depthDescriptorSet,
                                    0, nullptr);
            vkCmdPushConstants(commandBuffer,
                               m_depthPipelineLayout,
                               VK_SHADER_STAGE_COMPUTE_BIT,
                               0,
                               sizeof(PushConstants),
                               &pushConstants);
            vkCmdDispatch(commandBuffer, 1, 1, 1);

            transitionImageLayout(commandBuffer,
                                  depthImage,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                  VK_IMAGE_ASPECT_DEPTH_BIT);
        }

        VkBufferMemoryBarrier hashReadyBarriers[2]{};
        uint32_t barrierCount = 1;

        hashReadyBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        hashReadyBarriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        hashReadyBarriers[0].dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        hashReadyBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hashReadyBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hashReadyBarriers[0].buffer = m_hashBuffer;
        hashReadyBarriers[0].offset = 0;
        hashReadyBarriers[0].size = sizeof(uint32_t) * 8u;

        if (hashDepth)
        {
            hashReadyBarriers[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            hashReadyBarriers[1].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            hashReadyBarriers[1].dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            hashReadyBarriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            hashReadyBarriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            hashReadyBarriers[1].buffer = m_depthHashBuffer;
            hashReadyBarriers[1].offset = 0;
            hashReadyBarriers[1].size = sizeof(uint32_t) * 8u;
            barrierCount = 2;
        }

        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT,
                             0,
                             0, nullptr,
                             barrierCount, hashReadyBarriers,
                             0, nullptr);

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
    frameHash.depthHash = 0u;
    frameHash.frameIndex = frameIndex;
    vkUnmapMemory(m_device, m_hashMemory);

    const TransferPhase previousPhase =
        PcieTransferTracker::instance().currentPhase();
    PcieTransferTracker::instance().setPhase(TransferPhase::TEST_ONLY);
    PCIE_RECORD_D2H("sha256_hash_readback", sizeof(uint32_t) * 8u);
    PcieTransferTracker::instance().setPhase(previousPhase);

    if (hashDepth)
    {
        void* mappedDepthData = nullptr;
        if (vkMapMemory(m_device,
                        m_depthHashMemory,
                        0,
                        sizeof(uint32_t) * 8u,
                        0,
                        &mappedDepthData) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to map HashProbe depth hash memory.");
        }

        const uint32_t* depthWords =
            static_cast<const uint32_t*>(mappedDepthData);
        frameHash.depthHash = depthWords[7];
        vkUnmapMemory(m_device, m_depthHashMemory);
    }

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

    const bool depthMeasured = hasDepthHashes(hashes);
    const FrameHash& baseline = hashes.front();

    std::fprintf(file, "[Frame %06u] Color SHA256: ", baseline.frameIndex);
    writeHashWords(file, baseline.colorHash);
    if (depthMeasured)
    {
        std::fprintf(file, "  Depth SHA256-32: %08X", baseline.depthHash);
    }
    std::fprintf(file, "\n");

    for (size_t i = 1; i < hashes.size(); ++i)
    {
        const bool colorMatch = colorHashesMatch(baseline, hashes[i]);
        const bool depthMatch = !depthMeasured || depthHashesMatch(baseline, hashes[i]);

        std::fprintf(file, "[Frame %06u] Color SHA256: ", hashes[i].frameIndex);
        writeHashWords(file, hashes[i].colorHash);
        if (depthMeasured)
        {
            std::fprintf(file, "  Depth SHA256-32: %08X", hashes[i].depthHash);
        }
        std::fprintf(file,
                     " %s\n",
                     combinedStatusLabel(colorMatch, depthMeasured, depthMatch));
    }

    std::fclose(file);

    const float colorConsistency =
        computeConsistencyExcludingBaseline(hashes, colorHashesMatch);
    if (!depthMeasured)
    {
        return colorConsistency;
    }

    const float depthConsistency =
        computeConsistencyExcludingBaseline(hashes, depthHashesMatch);
    return (colorConsistency < depthConsistency) ? colorConsistency : depthConsistency;
}

void HashProbe::generateRootCauseReport(const std::vector<FrameHash>& hashes,
                                        const char* reportPath) const
{
    // TODO(v1.2): after 3DGS alpha blending lands, extend rootcause to
    //             report specific atomicAdd call sites in alpha_composite.comp
    const int totalFrames = static_cast<int>(hashes.size());
    const bool depthMeasured = hasDepthHashes(hashes);

    const int colorConsistentFrames =
        countMatchingFramesIncludingBaseline(hashes, colorHashesMatch);
    const int depthConsistentFrames = depthMeasured
        ? countMatchingFramesIncludingBaseline(hashes, depthHashesMatch)
        : 0;

    std::vector<uint32_t> colorDivergedFrames;
    std::vector<uint32_t> depthDivergedFrames;
    countMatchingFramesExcludingBaseline(hashes,
                                         colorHashesMatch,
                                         &colorDivergedFrames);
    if (depthMeasured)
    {
        countMatchingFramesExcludingBaseline(hashes,
                                             depthHashesMatch,
                                             &depthDivergedFrames);
    }

    const float colorConsistency = totalFrames == 0
        ? 0.0f
        : (100.0f * static_cast<float>(colorConsistentFrames) /
           static_cast<float>(totalFrames));
    const float depthConsistency = (!depthMeasured || totalFrames == 0)
        ? 0.0f
        : (100.0f * static_cast<float>(depthConsistentFrames) /
           static_cast<float>(totalFrames));

    const char* conclusionState = "pass";
    const char* conclusionDetail =
        "point-sprite path is bit-stable for the measured tensors.";

    if (totalFrames == 0)
    {
        conclusionState = "fail";
        conclusionDetail = "no frames were captured; the measurement is invalid.";
    }
    else if (colorConsistentFrames == 0 ||
             (depthMeasured && depthConsistentFrames == 0))
    {
        conclusionState = "fail";
        conclusionDetail =
            "at least one measured tensor diverged on every sampled frame.";
    }
    else if (colorConsistentFrames != totalFrames ||
             (depthMeasured && depthConsistentFrames != totalFrames))
    {
        conclusionState = "warn";
        conclusionDetail =
            "the current point-sprite path shows drift in at least one measured tensor.";
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

    std::fprintf(file, "# SHA-256 Determinism Analysis Report\n\n");

    std::fprintf(file, "## Known Non-Determinism Sources (point-sprite path)\n");
    std::fprintf(file,
                 "- NONE: current path uses no atomicAdd, no cross-warp reduction.\n");
    std::fprintf(file,
                 "- Shader: shaders/point.vert:14-18 - per-vertex transform only; no shared or global accumulation.\n");
    std::fprintf(file,
                 "- Shader: shaders/point.frag:6-14 - single outColor write at line 13, no parallel accumulation, no atomicAdd, no imageStore.\n");
    std::fprintf(file,
                 "- Determinism guarantee: valid ONLY for point-sprite pipeline.\n\n");

    std::fprintf(file, "## Expected Non-Determinism Sources (future 3DGS path)\n");
    std::fprintf(file,
                 "- shaders/alpha_composite.comp: line N/A (planned, not yet implemented)\n");
    std::fprintf(file,
                 "  Risk: atomicAdd on RGB accumulator across warps\n");
    std::fprintf(file,
                 "  -> floatNonAssociativity, see IEEE 754 sec. 5.9\n");
    std::fprintf(file,
                 "- shaders/radix_sort.comp: line N/A (planned, not yet implemented)\n");
    std::fprintf(file,
                 "  Risk: tie-breaking undefined for equal depth keys\n");
    std::fprintf(file,
                 "  -> GaussianData.id must be used as tiebreaker\n\n");

    std::fprintf(file, "## Measurement result\n");
    // TODO(v1.2): upgrade depth reporting from lower-32-bit summary to the
    //             full 256-bit digest once the 3DGS depth tensor is finalized.
    std::fprintf(file,
                 "- colorConsistency: %.2f%% (%d/%d frames identical)\n",
                 colorConsistency,
                 colorConsistentFrames,
                 totalFrames);
    if (depthMeasured)
    {
        std::fprintf(file,
                     "- depthConsistency: %.2f%% (%d/%d frames identical)\n",
                     depthConsistency,
                     depthConsistentFrames,
                     totalFrames);
    }
    else
    {
        std::fprintf(file,
                     "- depthConsistency: N/A (depth hash disabled or not supplied)\n");
    }

    std::fprintf(file, "- colorDivergedFrames: ");
    if (colorDivergedFrames.empty())
    {
        std::fprintf(file, "none\n");
    }
    else
    {
        for (size_t i = 0; i < colorDivergedFrames.size(); ++i)
        {
            std::fprintf(file,
                         "%s%06u",
                         (i == 0) ? "" : ", ",
                         colorDivergedFrames[i]);
        }
        std::fprintf(file, "\n");
    }

    std::fprintf(file, "- depthDivergedFrames: ");
    if (!depthMeasured)
    {
        std::fprintf(file, "N/A\n");
    }
    else if (depthDivergedFrames.empty())
    {
        std::fprintf(file, "none\n");
    }
    else
    {
        for (size_t i = 0; i < depthDivergedFrames.size(); ++i)
        {
            std::fprintf(file,
                         "%s%06u",
                         (i == 0) ? "" : ", ",
                         depthDivergedFrames[i]);
        }
        std::fprintf(file, "\n");
    }

    std::fprintf(file,
                 "- Conclusion: %s - %s\n\n",
                 conclusionState,
                 conclusionDetail);
    std::fprintf(file,
                 "Rerun required after: v1.2 (alpha blending), v2.0 (GPU radix sort)\n");

    std::fclose(file);
}

void HashProbe::transitionImageLayout(VkCommandBuffer cmd,
                                      VkImage image,
                                      VkImageLayout oldLayout,
                                      VkImageLayout newLayout,
                                      VkImageAspectFlags aspectMask)
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
    barrier.subresourceRange.aspectMask = aspectMask;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags destinationStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
        aspectMask == VK_IMAGE_ASPECT_COLOR_BIT)
    {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL &&
             aspectMask == VK_IMAGE_ASPECT_COLOR_BIT)
    {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL &&
             newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             aspectMask == VK_IMAGE_ASPECT_COLOR_BIT)
    {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             newLayout == VK_IMAGE_LAYOUT_GENERAL &&
             aspectMask == VK_IMAGE_ASPECT_COLOR_BIT)
    {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_GENERAL &&
             newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             aspectMask == VK_IMAGE_ASPECT_COLOR_BIT)
    {
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL &&
             newLayout == VK_IMAGE_LAYOUT_GENERAL &&
             aspectMask == VK_IMAGE_ASPECT_COLOR_BIT)
    {
        barrier.srcAccessMask =
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_GENERAL &&
             newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL &&
             aspectMask == VK_IMAGE_ASPECT_COLOR_BIT)
    {
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask =
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        destinationStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL &&
             newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
             aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT)
    {
        barrier.srcAccessMask =
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                      VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
             newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL &&
             aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT)
    {
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask =
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                           VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
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

    if (m_depthHashMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(m_device, m_depthHashMemory, nullptr);
        m_depthHashMemory = VK_NULL_HANDLE;
    }
    if (m_depthHashBuffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(m_device, m_depthHashBuffer, nullptr);
        m_depthHashBuffer = VK_NULL_HANDLE;
    }
    if (m_depthSampler != VK_NULL_HANDLE)
    {
        vkDestroySampler(m_device, m_depthSampler, nullptr);
        m_depthSampler = VK_NULL_HANDLE;
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
    if (m_depthDescriptorPool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(m_device, m_depthDescriptorPool, nullptr);
        m_depthDescriptorPool = VK_NULL_HANDLE;
    }
    m_depthDescriptorSet = VK_NULL_HANDLE;
    if (m_descriptorPool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
    }
    m_descriptorSet = VK_NULL_HANDLE;
    if (m_depthPipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(m_device, m_depthPipeline, nullptr);
        m_depthPipeline = VK_NULL_HANDLE;
    }
    if (m_pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(m_device, m_pipeline, nullptr);
        m_pipeline = VK_NULL_HANDLE;
    }
    if (m_depthPipelineLayout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(m_device, m_depthPipelineLayout, nullptr);
        m_depthPipelineLayout = VK_NULL_HANDLE;
    }
    if (m_pipelineLayout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
    }
    if (m_depthDescriptorLayout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(m_device, m_depthDescriptorLayout, nullptr);
        m_depthDescriptorLayout = VK_NULL_HANDLE;
    }
    if (m_descriptorLayout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(m_device, m_descriptorLayout, nullptr);
        m_descriptorLayout = VK_NULL_HANDLE;
    }

    m_depthImageView = VK_NULL_HANDLE;
    m_height = 0;
    m_width = 0;
    m_commandPool = VK_NULL_HANDLE;
    m_computeQueue = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
}

} // namespace SplatCore
