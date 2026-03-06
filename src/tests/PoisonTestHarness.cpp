#include "PoisonTestHarness.h"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace SplatCore {

namespace {

struct PushConstants {
    uint32_t poisonPattern;
    uint32_t elementCount;
};

static_assert(sizeof(PushConstants) == 8,
              "PushConstants must match the shader push_constant layout.");

} // namespace

std::vector<uint32_t> PoisonTestHarness::loadSpv(const char* path)
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
    const size_t readBytes = std::fread(code.data(), 1, static_cast<size_t>(fileSize), file);
    std::fclose(file);

    if (readBytes != static_cast<size_t>(fileSize))
    {
        std::terminate();
    }

    return code;
}

void PoisonTestHarness::init(VkDevice device,
                             VkPhysicalDevice /*physicalDevice*/,
                             VkCommandPool commandPool,
                             VkQueue computeQueue,
                             const char* spvPath)
{
    shutdown();

    m_device = device;
    m_commandPool = commandPool;
    m_computeQueue = computeQueue;

    const std::vector<uint32_t> shaderCode = loadSpv(spvPath);

    VkShaderModuleCreateInfo shaderModuleInfo{};
    shaderModuleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleInfo.codeSize = shaderCode.size() * sizeof(uint32_t);
    shaderModuleInfo.pCode = shaderCode.data();

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(m_device, &shaderModuleInfo, nullptr, &shaderModule) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create poison shader module.");
    }

    try
    {
        VkDescriptorSetLayoutBinding bufferBinding{};
        bufferBinding.binding = 0;
        bufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bufferBinding.descriptorCount = 1;
        bufferBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &bufferBinding;

        if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr,
                                        &m_descriptorSetLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create poison descriptor set layout.");
        }

        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

        if (vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr,
                                   &m_pipelineLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create poison pipeline layout.");
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

        if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                     nullptr, &m_pipeline) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create poison compute pipeline.");
        }

        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 16;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        poolInfo.maxSets = 16;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;

        if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr,
                                   &m_descriptorPool) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create poison descriptor pool.");
        }
    }
    catch (...)
    {
        vkDestroyShaderModule(m_device, shaderModule, nullptr);
        shutdown();
        throw;
    }

    vkDestroyShaderModule(m_device, shaderModule, nullptr);
}

void PoisonTestHarness::poisonBuffer(VkBuffer targetBuffer,
                                     VkDeviceSize bufferSize,
                                     PoisonPattern pattern)
{
    if (!isReady())
    {
        throw std::runtime_error("PoisonTestHarness is not initialized.");
    }
    if (targetBuffer == VK_NULL_HANDLE)
    {
        throw std::runtime_error("Poison target buffer is null.");
    }
    if (bufferSize == 0)
    {
        return;
    }
    if ((bufferSize % sizeof(uint32_t)) != 0)
    {
        std::terminate();
    }

    const VkDeviceSize elementCount64 = bufferSize / sizeof(uint32_t);
    if (elementCount64 > std::numeric_limits<uint32_t>::max())
    {
        std::terminate();
    }

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_descriptorSetLayout;

    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(m_device, &allocInfo, &descriptorSet) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate poison descriptor set.");
    }

    try
    {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = targetBuffer;
        bufferInfo.offset = 0;
        bufferInfo.range = bufferSize;

        VkWriteDescriptorSet descriptorWrite{};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = descriptorSet;
        descriptorWrite.dstBinding = 0;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(m_device, 1, &descriptorWrite, 0, nullptr);

        VkCommandBufferAllocateInfo commandAllocInfo{};
        commandAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandAllocInfo.commandPool = m_commandPool;
        commandAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandAllocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(m_device, &commandAllocInfo, &commandBuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate poison command buffer.");
        }

        try
        {
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to begin poison command buffer.");
            }

            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
            vkCmdBindDescriptorSets(commandBuffer,
                                    VK_PIPELINE_BIND_POINT_COMPUTE,
                                    m_pipelineLayout,
                                    0, 1, &descriptorSet,
                                    0, nullptr);

            const PushConstants pc{
                static_cast<uint32_t>(pattern),
                static_cast<uint32_t>(elementCount64)};
            vkCmdPushConstants(commandBuffer, m_pipelineLayout,
                               VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(PushConstants), &pc);

            const uint32_t groupCount = (pc.elementCount + 255u) / 256u;
            vkCmdDispatch(commandBuffer, groupCount, 1, 1);
            insertMemoryBarrier(commandBuffer);

            if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to end poison command buffer.");
            }

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;

            if (vkQueueSubmit(m_computeQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to submit poison command buffer.");
            }
            if (vkQueueWaitIdle(m_computeQueue) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed waiting for poison command buffer.");
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
        vkFreeDescriptorSets(m_device, m_descriptorPool, 1, &descriptorSet);
        throw;
    }

    if (vkFreeDescriptorSets(m_device, m_descriptorPool, 1, &descriptorSet) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to free poison descriptor set.");
    }
}

void PoisonTestHarness::insertMemoryBarrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask =
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0,
                         1, &barrier,
                         0, nullptr,
                         0, nullptr);
}

void PoisonTestHarness::shutdown()
{
    if (m_device == VK_NULL_HANDLE)
    {
        return;
    }

    if (m_descriptorPool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
    }
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
    if (m_descriptorSetLayout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
        m_descriptorSetLayout = VK_NULL_HANDLE;
    }

    m_commandPool = VK_NULL_HANDLE;
    m_computeQueue = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
}

} // namespace SplatCore
