#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/memory/MemorySystem.h"
#include "src/core/memory/VramLogger.h"
#include "src/tests/HashProbe.h"

using SplatCore::Allocation;
using SplatCore::AllocationDesc;
using SplatCore::MemoryRegion;
using SplatCore::MemorySystem;
using SplatCore::VramLogger;

// ── Vertex layout ────────────────────────────────────────────────────────
struct Vertex
{
    glm::vec3 pos;
    glm::vec3 color;

    static VkVertexInputBindingDescription getBindingDescription()
    {
        VkVertexInputBindingDescription binding{};
        binding.binding = 0;
        binding.stride = sizeof(Vertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return binding;
    }

    static std::array<VkVertexInputAttributeDescription, 2>
    getAttributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription, 2> attrs{};
        attrs[0].binding = 0;
        attrs[0].location = 0;
        attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[0].offset = offsetof(Vertex, pos);
        attrs[1].binding = 0;
        attrs[1].location = 1;
        attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[1].offset = offsetof(Vertex, color);
        return attrs;
    }
};

// ── UBO layout (matches cube.vert binding 0) ─────────────────────────────
struct UniformBufferObject
{
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

// ─────────────────────────────────────────────────────────────────────────

// ── FPS Camera ───────────────────────────────────────────────────────────
struct Camera
{
    glm::vec3 position = {0.0f, 0.0f, 3.0f};
    float yaw = -90.0f;        // degrees, -90 = facing -Z
    float pitch = 0.0f;        // degrees
    float speed = 5.0f;        // world units / second
    float sensitivity = 0.1f;  // degrees / pixel
    glm::vec3 worldUp = {0.0f, 1.0f, 0.0f};

    glm::vec3 getFront() const
    {
        glm::vec3 f;
        f.x = std::cos(glm::radians(yaw)) * std::cos(glm::radians(pitch));
        f.y = std::sin(glm::radians(pitch));
        f.z = std::sin(glm::radians(yaw)) * std::cos(glm::radians(pitch));
        return glm::normalize(f);
    }

    glm::mat4 getView() const
    {
        const glm::vec3 front = getFront();
        return glm::lookAt(position, position + front, worldUp);
    }

    void processKeyboard(GLFWwindow *window, float deltaTime)
    {
        const glm::vec3 front = getFront();
        const glm::vec3 right = glm::normalize(glm::cross(front, worldUp));
        const float v = speed * deltaTime;

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            position += front * v;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            position -= front * v;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            position -= right * v;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            position += right * v;
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
            position -= worldUp * v;
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
            position += worldUp * v;
    }

    // Apply mouse delta (pixels). Y is pre-inverted by caller.
    void processMouse(float dx, float dy)
    {
        yaw += dx * sensitivity;
        pitch = glm::clamp(pitch + dy * sensitivity, -89.0f, 89.0f);
    }
};
// ─────────────────────────────────────────────────────────────────────────

class SplatCoreApp
{
public:
    ~SplatCoreApp();
    void run();
    void setPlyPath(const std::string &path) { plyFilePath = path; }
    void initializeWindowForTesting();
    void initializeVulkanCoreForTesting();
    void initializeRenderResourcesForTesting();
    void renderFramesForTesting(uint32_t frameCount);
    void readbackOffscreenForTesting(std::vector<uint32_t> &outPixels);
    void shutdownForTesting();
    void enableHashProbeForTesting(const std::string &spvPath);
    const std::vector<SplatCore::FrameHash> &frameHashesForTesting() const
    {
        return frameHashes;
    }
    VkDevice deviceForTesting() const { return device; }
    VkPhysicalDevice physicalDeviceForTesting() const { return physicalDevice; }
    VkCommandPool commandPoolForTesting() const { return commandPool; }
    VkQueue computeQueueForTesting() const { return graphicsQueue; }
    VkBuffer offscreenReadbackBufferForTesting() const { return offscreenReadbackBuffer; }
    VkDeviceSize offscreenReadbackBufferSizeForTesting() const
    {
        return offscreenReadbackAllocation.size;
    }
    VkPhysicalDeviceProperties physicalDevicePropertiesForTesting() const;

private:
    static constexpr uint32_t kWidth = 800;
    static constexpr uint32_t kHeight = 600;
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
    static constexpr const char *kWindowTitle = "SplatCore v0.5";

#ifdef VK_ENABLE_VALIDATION_LAYERS
    const std::vector<const char *> validationLayers = {
        "VK_LAYER_KHRONOS_validation"};
#else
    const std::vector<const char *> validationLayers; // empty in Release
#endif

    const std::vector<const char *> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    struct QueueFamilyIndices
    {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;

        bool isComplete() const
        {
            return graphicsFamily.has_value() && presentFamily.has_value();
        }
    };

    struct SwapChainSupportDetails
    {
        VkSurfaceCapabilitiesKHR capabilities{};
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };

    GLFWwindow *window = nullptr;
    bool glfwInitialized = false;
    bool cleanedUp = false;

    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue presentQueue = VK_NULL_HANDLE;
    VkSwapchainKHR swapChain = VK_NULL_HANDLE;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D swapChainExtent{};
    std::vector<VkImageView> swapChainImageViews;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    // Depth buffer (tied to swapchain extent)
    VkImage depthImage = VK_NULL_HANDLE;
    Allocation depthImageAllocation{};
    VkImageView depthImageView = VK_NULL_HANDLE;
    // Offscreen Render Target（v0.6 毒化测试用）
    VkImage offscreenImage = VK_NULL_HANDLE;
    VkDeviceMemory offscreenMemory = VK_NULL_HANDLE; // 保留备用
    VkImageView offscreenImageView = VK_NULL_HANDLE;
    VkFramebuffer offscreenFramebuffer = VK_NULL_HANDLE;
    SplatCore::Allocation offscreenAllocation{};
    VkBuffer offscreenReadbackBuffer = VK_NULL_HANDLE;
    SplatCore::Allocation offscreenReadbackAllocation{};

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;

    // Point cloud geometry buffer
    VkBuffer pointVertexBuffer = VK_NULL_HANDLE;
    Allocation pointVertexBufferAllocation{};
    uint32_t pointCount = 0;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<Allocation> uniformBufferAllocations;
    std::vector<void *> uniformBuffersMapped;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets;
    // Stores the currentFrame index (0..MAX_FRAMES_IN_FLIGHT-1)
    // that last submitted to this swapchain image, or kNoFrame.
    static constexpr uint32_t kNoFrame = UINT32_MAX;
    std::vector<uint32_t> imagesInFlight;
    uint32_t currentFrame = 0;
    bool framebufferResized = false;
    double lastFpsTime = 0.0;
    uint32_t frameCount = 0;
    uint32_t frameIndex = 1;
    uint32_t maxFrameCount = 0;
    double lastFrameTime = 0.0;
    bool hashProbeRequested = false;
    std::string hashProbeSpvPath;
    SplatCore::HashProbe hashProbe;
    std::vector<SplatCore::FrameHash> frameHashes;
    bool hashProbeEnabled = false;

    // FPS camera and mouse state
    Camera camera{};
    bool mouseCaptured = false;
    bool firstMouse = true;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;

    // PLY file path — set from argv before run()
    std::string plyFilePath;

    std::atomic<bool> validationErrorDetected{false};
    std::string validationErrorMessage;

    void initWindow();
    void initVulkan();
    void initVulkanCore();
    void initRenderResources();
    void mainLoop();
    void cleanup();
    void drawFrame();
    void createInstance();
    void setupDebugMessenger();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapChain();
    void createImageViews();
    void createRenderPass();
    void createFramebuffers();
    void createCommandPool();
    void createCommandBuffer();
    void createSyncObjects();
    void createOffscreenTarget();
    void destroyOffscreenTarget();
    void readbackOffscreenFrame(std::vector<uint32_t> &outPixels);
    void cleanupSwapChain();
    void recreateSwapChain();
    void createDepthResources();
    void createDescriptorSetLayout();
    void createGraphicsPipeline();
    void loadPointCloud();
    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();
    void updateUniformBuffer(uint32_t frameSlot);
    uint32_t findMemoryType(uint32_t typeFilter,
                            VkMemoryPropertyFlags properties) const;
    void createBuffer(VkDeviceSize size,
                      VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties,
                      VkBuffer &buffer,
                      Allocation &allocation,
                      MemoryRegion region,
                      std::string_view allocationName);
    void destroyBufferAllocation(VkBuffer &buffer,
                                 Allocation &allocation);
    void copyBuffer(VkBuffer srcBuffer,
                    VkBuffer dstBuffer,
                    VkDeviceSize size);
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);
    void createImage(uint32_t width, uint32_t height,
                     VkFormat format, VkImageTiling tiling,
                     VkImageUsageFlags usage,
                     VkMemoryPropertyFlags properties,
                     VkImage &image,
                     Allocation &allocation,
                     MemoryRegion region,
                     std::string_view allocationName);
    void destroyImageAllocation(VkImage &image,
                                Allocation &allocation);
    VkImageView createImageView(VkImage image, VkFormat format,
                                VkImageAspectFlags aspectFlags);
    VkFormat findSupportedFormat(const std::vector<VkFormat> &candidates,
                                 VkImageTiling tiling,
                                 VkFormatFeatureFlags features) const;
    VkFormat findDepthFormat() const;
    static void framebufferResizeCallback(GLFWwindow *window,
                                          int width, int height);
    static void cursorPosCallback(GLFWwindow *window,
                                  double xpos, double ypos);
    static void mouseButtonCallback(GLFWwindow *window,
                                    int button, int action, int mods);
    static void scrollCallback(GLFWwindow *window,
                               double xoffset, double yoffset);
    void recordCommandBuffer(VkCommandBuffer cmdBuffer, uint32_t imageIndex);
    static std::vector<char> readSpvFile(const std::string &filename);
    VkShaderModule createShaderModule(const std::vector<char> &code);
    bool checkValidationLayerSupport() const;
    std::vector<const char *> getRequiredExtensions() const;
    bool isDeviceSuitable(VkPhysicalDevice candidateDevice) const;
    bool checkDeviceExtensionSupport(VkPhysicalDevice candidateDevice) const;
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice candidateDevice) const;
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice candidateDevice) const;
    static VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats);
    static VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities) const;
    void failIfValidationIssueDetected() const;
    static void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo);
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
        void *pUserData);
    static VkResult createDebugUtilsMessengerEXT(
        VkInstance vkInstance,
        const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
        const VkAllocationCallbacks *pAllocator,
        VkDebugUtilsMessengerEXT *pDebugMessenger);
    static void destroyDebugUtilsMessengerEXT(
        VkInstance vkInstance,
        VkDebugUtilsMessengerEXT vkDebugMessenger,
        const VkAllocationCallbacks *pAllocator);
};

SplatCoreApp::~SplatCoreApp()
{
    cleanup();
}

void SplatCoreApp::run()
{
    char* envValue = nullptr;
    size_t envValueLength = 0;
    if (_dupenv_s(&envValue, &envValueLength, "SPLATCORE_MAX_FRAMES") == 0 &&
        envValue != nullptr)
    {
        const unsigned long parsedValue = std::strtoul(envValue, nullptr, 10);
        if (parsedValue > 0)
        {
            maxFrameCount = static_cast<uint32_t>(parsedValue);
        }
    }
    std::free(envValue);

    initWindow();
    initVulkan();
    mainLoop();
}

void SplatCoreApp::initializeWindowForTesting()
{
    initWindow();
}

void SplatCoreApp::initializeVulkanCoreForTesting()
{
    initVulkanCore();
}

void SplatCoreApp::initializeRenderResourcesForTesting()
{
    initRenderResources();
}

void SplatCoreApp::renderFramesForTesting(uint32_t targetFrameCount)
{
    maxFrameCount = targetFrameCount;
    mainLoop();
}

void SplatCoreApp::readbackOffscreenForTesting(std::vector<uint32_t> &outPixels)
{
    if (device != VK_NULL_HANDLE && vkDeviceWaitIdle(device) != VK_SUCCESS)
    {
        throw std::runtime_error("vkDeviceWaitIdle failed before offscreen readback.");
    }
    readbackOffscreenFrame(outPixels);
    failIfValidationIssueDetected();
}

void SplatCoreApp::shutdownForTesting()
{
    cleanup();
}

void SplatCoreApp::enableHashProbeForTesting(const std::string &spvPath)
{
    hashProbeRequested = true;
    hashProbeSpvPath = spvPath;
}

VkPhysicalDeviceProperties SplatCoreApp::physicalDevicePropertiesForTesting() const
{
    VkPhysicalDeviceProperties props{};
    if (physicalDevice != VK_NULL_HANDLE)
    {
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
    }
    return props;
}

void SplatCoreApp::initWindow()
{
    if (glfwInit() == GLFW_FALSE)
    {
        throw std::runtime_error("Failed to initialize GLFW.");
    }
    glfwInitialized = true;

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_DECORATED, GLFW_TRUE);
    glfwWindowHint(GLFW_FLOATING, GLFW_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(static_cast<int>(kWidth), static_cast<int>(kHeight), kWindowTitle, nullptr, nullptr);
    if (window == nullptr)
    {
        throw std::runtime_error("Failed to create GLFW window.");
    }
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);
    lastFpsTime = glfwGetTime();
    lastFrameTime = glfwGetTime();
}

void SplatCoreApp::initVulkan()
{
    initVulkanCore();
    initRenderResources();
}

void SplatCoreApp::initVulkanCore()
{
    createInstance();
#ifdef VK_ENABLE_VALIDATION_LAYERS
    setupDebugMessenger();
#endif
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    MemorySystem::init(instance, physicalDevice, device);
    VramLogger::init("engine.log");
    createCommandPool();
}

void SplatCoreApp::initRenderResources()
{
    createSwapChain();
    createImageViews();
    createDepthResources(); // depth image sized to swapchain extent
    createRenderPass();     // renderPass references depth format
    createDescriptorSetLayout(); // must exist before pipeline
    createGraphicsPipeline();
    createFramebuffers();
    loadPointCloud(); // parse PLY + staging upload to GPU
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffer();
    createSyncObjects();
    createOffscreenTarget();
    failIfValidationIssueDetected();

    const char* hashProbeEnv = nullptr;
#pragma warning(push)
#pragma warning(disable: 4996)
    hashProbeEnv = std::getenv("SPLATCORE_HASH_PROBE");
#pragma warning(pop)
    const bool enableHashProbeFromEnv =
        hashProbeEnv != nullptr && std::string(hashProbeEnv) == "1";
    if (enableHashProbeFromEnv || hashProbeRequested)
    {
        if (hashProbe.isReady())
        {
            hashProbe.shutdown();
        }

        const std::string hashProbeShaderPath =
            hashProbeSpvPath.empty()
                ? std::string(SHADER_DIR) + "sha256_compute.spv"
                : hashProbeSpvPath;
        hashProbe.init(device,
                       physicalDevice,
                       commandPool,
                       graphicsQueue,
                       offscreenImageView,
                       swapChainExtent.width,
                       swapChainExtent.height,
                       hashProbeShaderPath.c_str());
        hashProbeEnabled = true;
        frameHashes.clear();
        frameHashes.reserve(100);
    }
}

void SplatCoreApp::mainLoop()
{
    while (glfwWindowShouldClose(window) == GLFW_FALSE)
    {
        drawFrame();
    }

    if (vkDeviceWaitIdle(device) != VK_SUCCESS)
    {
        throw std::runtime_error("vkDeviceWaitIdle failed.");
    }
}

void SplatCoreApp::cleanupSwapChain()
{
    destroyOffscreenTarget();

    // Depth buffer is tied to swapchain extent — destroy before framebuffers.
    if (depthImageView != VK_NULL_HANDLE)
    {
        vkDestroyImageView(device, depthImageView, nullptr);
        depthImageView = VK_NULL_HANDLE;
    }
    destroyImageAllocation(depthImage, depthImageAllocation);

    for (VkFramebuffer framebuffer : swapChainFramebuffers)
    {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }
    swapChainFramebuffers.clear();

    for (VkImageView imageView : swapChainImageViews)
    {
        vkDestroyImageView(device, imageView, nullptr);
    }
    swapChainImageViews.clear();

    if (swapChain != VK_NULL_HANDLE)
    {
        vkDestroySwapchainKHR(device, swapChain, nullptr);
        swapChain = VK_NULL_HANDLE;
    }
}

void SplatCoreApp::recreateSwapChain()
{
    // Handle minimize: wait until window has non-zero size.
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0)
    {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(device);

    cleanupSwapChain();

    createSwapChain();
    createImageViews();
    createDepthResources(); // must come before createFramebuffers
    createFramebuffers();
    createOffscreenTarget();

    if (hashProbeEnabled && hashProbe.isReady())
    {
        hashProbe.shutdown();

        const std::string hashProbeShaderPath =
            hashProbeSpvPath.empty()
                ? std::string(SHADER_DIR) + "sha256_compute.spv"
                : hashProbeSpvPath;
        hashProbe.init(device,
                       physicalDevice,
                       commandPool,
                       graphicsQueue,
                       offscreenImageView,
                       swapChainExtent.width,
                       swapChainExtent.height,
                       hashProbeShaderPath.c_str());
        frameHashes.clear();
        frameHashes.reserve(100);
    }

    // renderFinishedSemaphores are sized by swapchain image count,
    // which may have changed — destroy old ones and recreate.
    for (VkSemaphore semaphore : renderFinishedSemaphores)
    {
        vkDestroySemaphore(device, semaphore, nullptr);
    }
    renderFinishedSemaphores.clear();

    renderFinishedSemaphores.resize(swapChainImages.size());
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (size_t i = 0; i < swapChainImages.size(); ++i)
    {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr,
                              &renderFinishedSemaphores[i]) != VK_SUCCESS)
        {
            throw std::runtime_error(
                "Failed to recreate renderFinished semaphore.");
        }
    }

    // imagesInFlight tracks per-image usage; reset for new image set.
    imagesInFlight.assign(swapChainImages.size(), kNoFrame);
}

void SplatCoreApp::framebufferResizeCallback(
    GLFWwindow *window, int /*width*/, int /*height*/)
{
    auto *app = reinterpret_cast<SplatCoreApp *>(
        glfwGetWindowUserPointer(window));
    if (app != nullptr)
    {
        app->framebufferResized = true;
    }
}

void SplatCoreApp::cleanup()
{
    if (cleanedUp)
    {
        return;
    }
    cleanedUp = true;

    if (device != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(device);
    }

    if (hashProbe.isReady())
    {
        hashProbe.shutdown();
    }

    for (VkSemaphore semaphore : imageAvailableSemaphores)
    {
        vkDestroySemaphore(device, semaphore, nullptr);
    }
    imageAvailableSemaphores.clear();

    for (VkSemaphore semaphore : renderFinishedSemaphores)
    {
        vkDestroySemaphore(device, semaphore, nullptr);
    }
    renderFinishedSemaphores.clear();

    for (VkFence fence : inFlightFences)
    {
        vkDestroyFence(device, fence, nullptr);
    }
    inFlightFences.clear();
    imagesInFlight.clear();

    if (commandPool != VK_NULL_HANDLE && !commandBuffers.empty())
    {
        vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
        commandBuffers.clear();
    }
    if (commandPool != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(device, commandPool, nullptr);
        commandPool = VK_NULL_HANDLE;
    }

    // Descriptor resources
    if (descriptorPool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        descriptorPool = VK_NULL_HANDLE;
    }
    if (descriptorSetLayout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        descriptorSetLayout = VK_NULL_HANDLE;
    }

    // Uniform buffers (persistently mapped — explicit unmap before free)
    for (size_t i = 0; i < uniformBuffers.size(); ++i)
    {
        if (uniformBuffersMapped[i] != nullptr &&
            uniformBufferAllocations[i].vmaAllocation != nullptr)
        {
            vmaUnmapMemory(MemorySystem::allocator(),
                           uniformBufferAllocations[i].vmaAllocation);
            uniformBuffersMapped[i] = nullptr;
        }
        destroyBufferAllocation(uniformBuffers[i], uniformBufferAllocations[i]);
    }
    uniformBuffers.clear();
    uniformBufferAllocations.clear();
    uniformBuffersMapped.clear();

    // Point cloud geometry buffer
    destroyBufferAllocation(pointVertexBuffer, pointVertexBufferAllocation);

    destroyOffscreenTarget();
    cleanupSwapChain(); // destroys framebuffers, imageViews, swapchain

    VramLogger::shutdown();
    MemorySystem::shutdown();

    if (graphicsPipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        graphicsPipeline = VK_NULL_HANDLE;
    }
    if (pipelineLayout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
    }
    if (renderPass != VK_NULL_HANDLE)
    {
        vkDestroyRenderPass(device, renderPass, nullptr);
        renderPass = VK_NULL_HANDLE;
    }

    if (device != VK_NULL_HANDLE)
    {
        vkDestroyDevice(device, nullptr);
        device = VK_NULL_HANDLE;
    }

    if (debugMessenger != VK_NULL_HANDLE)
    {
        destroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        debugMessenger = VK_NULL_HANDLE;
    }

    if (surface != VK_NULL_HANDLE)
    {
        vkDestroySurfaceKHR(instance, surface, nullptr);
        surface = VK_NULL_HANDLE;
    }

    if (instance != VK_NULL_HANDLE)
    {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }

    if (window != nullptr)
    {
        glfwDestroyWindow(window);
        window = nullptr;
    }

    if (glfwInitialized)
    {
        glfwTerminate();
        glfwInitialized = false;
    }
}

void SplatCoreApp::drawFrame()
{
    failIfValidationIssueDetected();

    // 0) Delta time — camera movement is frame-rate independent.
    const double now = glfwGetTime();
    const float deltaTime = static_cast<float>(now - lastFrameTime);
    lastFrameTime = now;

    // Keyboard: WASD movement
    camera.processKeyboard(window, deltaTime);

    // ESC: first press releases mouse, second press closes window
    static bool escWasPressed = false;
    static bool escCloseArmed = false;
    const bool escNow = (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS);
    if (escNow && !escWasPressed)
    {
        if (!escCloseArmed)
        {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            mouseCaptured = false;
            escCloseArmed = true;
        }
        else
        {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
    }
    escWasPressed = escNow;

    // 1) CPU throttle: wait for this frame slot to be free.
    //    Do NOT reset the fence yet — acquire might fail.
    if (vkWaitForFences(device, 1, &inFlightFences[currentFrame],
                        VK_TRUE, UINT64_MAX) != VK_SUCCESS)
    {
        throw std::runtime_error("vkWaitForFences failed.");
    }

    // 2) Acquire swapchain image.
    uint32_t imageIndex = 0;
    const VkResult acquireResult = vkAcquireNextImageKHR(
        device,
        swapChain,
        UINT64_MAX,
        imageAvailableSemaphores[currentFrame],
        VK_NULL_HANDLE,
        &imageIndex);

    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR)
    {
        // Fence was NOT reset — safe to recreate and return.
        recreateSwapChain();
        glfwPollEvents();
        return;
    }
    if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR)
    {
        throw std::runtime_error("vkAcquireNextImageKHR failed.");
    }

    // 3) Acquire succeeded — NOW safe to reset the fence.
    if (vkResetFences(device, 1, &inFlightFences[currentFrame]) != VK_SUCCESS)
    {
        throw std::runtime_error("vkResetFences failed.");
    }

    // 4) Per-image fence tracking.
    {
        const uint32_t prevFrame = imagesInFlight[imageIndex];
        if (prevFrame != kNoFrame && prevFrame != currentFrame)
        {
            if (vkWaitForFences(device, 1, &inFlightFences[prevFrame],
                                VK_TRUE, UINT64_MAX) != VK_SUCCESS)
            {
                throw std::runtime_error(
                    "vkWaitForFences failed for imagesInFlight[imageIndex].");
            }
        }
        imagesInFlight[imageIndex] = currentFrame;
    }

    // 5a) Update MVP matrix for this frame.
    updateUniformBuffer(currentFrame);

    // 5) Record command buffer.
    VkCommandBuffer currentCommandBuffer = commandBuffers[currentFrame];
    if (vkResetCommandBuffer(currentCommandBuffer, 0) != VK_SUCCESS)
    {
        throw std::runtime_error("vkResetCommandBuffer failed.");
    }
    recordCommandBuffer(currentCommandBuffer, imageIndex);

    // 6) Submit.
    const VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
    const VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    const VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[imageIndex]};

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &currentCommandBuffer;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo,
                      inFlightFences[currentFrame]) != VK_SUCCESS)
    {
        throw std::runtime_error("vkQueueSubmit failed.");
    }

    if (hashProbeEnabled && hashProbe.isReady())
    {
        const auto hash = hashProbe.computeHash(offscreenImage, frameIndex);
        frameHashes.push_back(hash);

        if (frameHashes.size() >= 100)
        {
            const float rate =
                hashProbe.analyzeConsistency(frameHashes, "sha256_log.txt");
            hashProbe.generateRootCauseReport(frameHashes, "sha256_rootcause.md");
            std::fprintf(stdout,
                         "[HashProbe] 100帧采集完成，一致率: %.1f%%\n",
                         rate * 100.0f);
            hashProbeEnabled = false;
        }
    }

    // 7) Present.
    const VkSwapchainKHR swapChains[] = {swapChain};
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;

    const VkResult presentResult = vkQueuePresentKHR(presentQueue, &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR ||
        presentResult == VK_SUBOPTIMAL_KHR ||
        framebufferResized)
    {
        framebufferResized = false;
        recreateSwapChain();
    }
    else if (presentResult != VK_SUCCESS)
    {
        throw std::runtime_error("vkQueuePresentKHR failed.");
    }

    // Present has completed for this frame. Before the next frame starts,
    // force-release DYNAMIC allocations, then snapshot and log VRAM usage.
    MemorySystem::flushDynamicAllocations();
    const auto snap = MemorySystem::snapshot(frameIndex);
    VramLogger::logFrame(snap);
    ++frameIndex;

    if (maxFrameCount != 0 && frameIndex > maxFrameCount)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

    // 8) FPS counter — update title every second.
    frameCount++;
    if (now - lastFpsTime >= 1.0)
    {
        const double fps = static_cast<double>(frameCount) / (now - lastFpsTime);
        const std::string title =
            std::string(kWindowTitle) +
            "  |  FPS: " + std::to_string(static_cast<int>(fps));
        glfwSetWindowTitle(window, title.c_str());
        frameCount = 0;
        lastFpsTime = now;
    }

    glfwPollEvents();
    failIfValidationIssueDetected();
}

void SplatCoreApp::createInstance()
{
#ifdef VK_ENABLE_VALIDATION_LAYERS
    if (!checkValidationLayerSupport())
    {
        throw std::runtime_error(
            "Validation layer VK_LAYER_KHRONOS_validation is not available.");
    }
#endif

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "SplatCore";
    appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    appInfo.pEngineName = "SplatCore Engine";
    appInfo.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    const std::vector<const char *> extensions = getRequiredExtensions();

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount =
        static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames =
        validationLayers.empty() ? nullptr : validationLayers.data();

#ifdef VK_ENABLE_VALIDATION_LAYERS
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    populateDebugMessengerCreateInfo(debugCreateInfo);
    debugCreateInfo.pUserData = this;

    // Enable BestPractices validation feature.
    // Chain: createInfo.pNext -> validationFeatures -> debugCreateInfo -> nullptr
    const VkValidationFeatureEnableEXT bestPracticesEnable =
        VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT;
    VkValidationFeaturesEXT validationFeatures{};
    validationFeatures.sType =
        VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
    validationFeatures.enabledValidationFeatureCount = 1;
    validationFeatures.pEnabledValidationFeatures = &bestPracticesEnable;
    validationFeatures.pNext = &debugCreateInfo;

    createInfo.pNext = &validationFeatures;
#endif

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create Vulkan instance.");
    }
}

void SplatCoreApp::setupDebugMessenger()
{
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    populateDebugMessengerCreateInfo(createInfo);
    createInfo.pUserData = this;

    if (createDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create debug messenger.");
    }
}

void SplatCoreApp::createSurface()
{
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create window surface.");
    }
}

void SplatCoreApp::pickPhysicalDevice()
{
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0)
    {
        throw std::runtime_error("No Vulkan-capable GPU found.");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    VkPhysicalDevice firstSuitableDevice = VK_NULL_HANDLE;

    for (VkPhysicalDevice candidate : devices)
    {
        if (!isDeviceSuitable(candidate))
        {
            continue;
        }

        if (firstSuitableDevice == VK_NULL_HANDLE)
        {
            firstSuitableDevice = candidate;
        }

        VkPhysicalDeviceProperties deviceProperties{};
        vkGetPhysicalDeviceProperties(candidate, &deviceProperties);
        if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            physicalDevice = candidate;
            break;
        }
    }

    if (physicalDevice == VK_NULL_HANDLE && firstSuitableDevice != VK_NULL_HANDLE)
    {
        physicalDevice = firstSuitableDevice;
    }

    if (physicalDevice == VK_NULL_HANDLE)
    {
        throw std::runtime_error("Failed to find a suitable GPU.");
    }
}

void SplatCoreApp::createLogicalDevice()
{
    const QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    std::set<uint32_t> uniqueQueueFamilies = {
        indices.graphicsFamily.value(),
        indices.presentFamily.value()};

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    queueCreateInfos.reserve(uniqueQueueFamilies.size());
    const float queuePriority = 1.0f;

    for (uint32_t queueFamily : uniqueQueueFamilies)
    {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures deviceFeatures{};
    vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();
    createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames =
        validationLayers.empty() ? nullptr : validationLayers.data();

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create logical device.");
    }

    vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
}

void SplatCoreApp::createSwapChain()
{
    const SwapChainSupportDetails supportDetails = querySwapChainSupport(physicalDevice);

    const VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(supportDetails.formats);
    const VkPresentModeKHR presentMode = chooseSwapPresentMode(supportDetails.presentModes);
    const VkExtent2D extent = chooseSwapExtent(supportDetails.capabilities);

    uint32_t imageCount = supportDetails.capabilities.minImageCount + 1;
    if (supportDetails.capabilities.maxImageCount > 0 && imageCount > supportDetails.capabilities.maxImageCount)
    {
        imageCount = supportDetails.capabilities.maxImageCount;
    }

    const QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    const uint32_t queueFamilyIndices[] = {
        indices.graphicsFamily.value(),
        indices.presentFamily.value()};

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    if (indices.graphicsFamily != indices.presentFamily)
    {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    else
    {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0;
        createInfo.pQueueFamilyIndices = nullptr;
    }

    createInfo.preTransform = supportDetails.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create swap chain.");
    }

    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;
}

void SplatCoreApp::createImageViews()
{
    swapChainImageViews.resize(swapChainImages.size());

    for (size_t i = 0; i < swapChainImages.size(); ++i)
    {
        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = swapChainImages[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = swapChainImageFormat;
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create image view.");
        }
    }
}

void SplatCoreApp::createRenderPass()
{
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapChainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = findDepthFormat();
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                               VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    const std::array<VkAttachmentDescription, 2> attachments = {
        colorAttachment, depthAttachment};

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create render pass.");
    }
}

void SplatCoreApp::createFramebuffers()
{
    swapChainFramebuffers.resize(swapChainImageViews.size());

    for (size_t i = 0; i < swapChainImageViews.size(); ++i)
    {
        const std::array<VkImageView, 2> attachments = {
            swapChainImageViews[i], depthImageView};

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        framebufferInfo.pAttachments = attachments.data();
        framebufferInfo.width = swapChainExtent.width;
        framebufferInfo.height = swapChainExtent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create framebuffer.");
        }
    }
}

void SplatCoreApp::createCommandPool()
{
    const QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create command pool.");
    }
}

void SplatCoreApp::createCommandBuffer()
{
    commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;

    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate command buffer.");
    }
}

void SplatCoreApp::createSyncObjects()
{
    // imageAvailable: one per frame-in-flight slot (indexed by currentFrame)
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    // renderFinished: one per swapchain image (indexed by imageIndex)
    // because the presentation engine holds this semaphore until display is done
    renderFinishedSemaphores.resize(swapChainImages.size());
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    imagesInFlight.resize(swapChainImages.size(), kNoFrame);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    // Loop A: imageAvailable semaphores — MAX_FRAMES_IN_FLIGHT count
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create imageAvailable semaphore.");
        }
    }

    // Loop B: renderFinished semaphores — one per swapchain image
    for (size_t i = 0; i < swapChainImages.size(); ++i)
    {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create renderFinished semaphore.");
        }
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        if (vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create frame fence.");
        }
    }
}

void SplatCoreApp::destroyOffscreenTarget()
{
    if (offscreenFramebuffer != VK_NULL_HANDLE)
    {
        vkDestroyFramebuffer(device, offscreenFramebuffer, nullptr);
        offscreenFramebuffer = VK_NULL_HANDLE;
    }
    if (offscreenImageView != VK_NULL_HANDLE)
    {
        vkDestroyImageView(device, offscreenImageView, nullptr);
        offscreenImageView = VK_NULL_HANDLE;
    }
    MemorySystem::free(offscreenAllocation);
    MemorySystem::free(offscreenReadbackAllocation);
    offscreenImage = VK_NULL_HANDLE;
    offscreenMemory = VK_NULL_HANDLE;
    offscreenReadbackBuffer = VK_NULL_HANDLE;
}

void SplatCoreApp::createOffscreenTarget()
{
    destroyOffscreenTarget();

    constexpr VkFormat offscreenHashFormat = VK_FORMAT_R8G8B8A8_UINT;

    createImage(swapChainExtent.width,
                swapChainExtent.height,
                offscreenHashFormat,
                VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_STORAGE_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                offscreenImage,
                offscreenAllocation,
                MemoryRegion::STATIC,
                "PoisonTest::OffscreenColorTarget");

    try
    {
        offscreenImageView = createImageView(offscreenImage,
                                             offscreenHashFormat,
                                             VK_IMAGE_ASPECT_COLOR_BIT);

        AllocationDesc readbackDesc{};
        readbackDesc.region = MemoryRegion::STATIC;
        readbackDesc.bufferUsage =
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        readbackDesc.vmaUsage = VMA_MEMORY_USAGE_CPU_ONLY;
        readbackDesc.size =
            static_cast<VkDeviceSize>(swapChainExtent.width) *
            static_cast<VkDeviceSize>(swapChainExtent.height) *
            sizeof(uint32_t);
        readbackDesc.allocationName = "PoisonTest::ReadbackBuffer";
        readbackDesc.imageInfo = nullptr;

        offscreenReadbackAllocation = MemorySystem::allocate(readbackDesc);
        offscreenReadbackBuffer = offscreenReadbackAllocation.buffer;

        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkImageMemoryBarrier toTransferDst{};
        toTransferDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toTransferDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toTransferDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toTransferDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransferDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransferDst.image = offscreenImage;
        toTransferDst.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toTransferDst.subresourceRange.baseMipLevel = 0;
        toTransferDst.subresourceRange.levelCount = 1;
        toTransferDst.subresourceRange.baseArrayLayer = 0;
        toTransferDst.subresourceRange.layerCount = 1;
        toTransferDst.srcAccessMask = 0;
        toTransferDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0,
                             0, nullptr,
                             0, nullptr,
                             1, &toTransferDst);

        endSingleTimeCommands(commandBuffer);
    }
    catch (...)
    {
        destroyOffscreenTarget();
        throw;
    }
}

void SplatCoreApp::readbackOffscreenFrame(std::vector<uint32_t> &outPixels)
{
    if (offscreenImage == VK_NULL_HANDLE ||
        offscreenReadbackBuffer == VK_NULL_HANDLE)
    {
        throw std::runtime_error("Offscreen target is not initialized.");
    }

    const size_t pixelCount =
        static_cast<size_t>(swapChainExtent.width) *
        static_cast<size_t>(swapChainExtent.height);
    const VkDeviceSize byteSize =
        static_cast<VkDeviceSize>(pixelCount) * sizeof(uint32_t);

    outPixels.resize(pixelCount);

    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkImageMemoryBarrier toTransferSrc{};
    toTransferSrc.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toTransferSrc.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toTransferSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toTransferSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferSrc.image = offscreenImage;
    toTransferSrc.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toTransferSrc.subresourceRange.baseMipLevel = 0;
    toTransferSrc.subresourceRange.levelCount = 1;
    toTransferSrc.subresourceRange.baseArrayLayer = 0;
    toTransferSrc.subresourceRange.layerCount = 1;
    toTransferSrc.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toTransferSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &toTransferSrc);

    VkBufferImageCopy copyRegion{};
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageExtent = {
        swapChainExtent.width,
        swapChainExtent.height,
        1};

    vkCmdCopyImageToBuffer(commandBuffer,
                           offscreenImage,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           offscreenReadbackBuffer,
                           1,
                           &copyRegion);

    VkBufferMemoryBarrier bufferToHost{};
    bufferToHost.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferToHost.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bufferToHost.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    bufferToHost.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferToHost.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferToHost.buffer = offscreenReadbackBuffer;
    bufferToHost.offset = 0;
    bufferToHost.size = byteSize;

    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         0,
                         0, nullptr,
                         1, &bufferToHost,
                         0, nullptr);

    VkImageMemoryBarrier backToColorAttachment{};
    backToColorAttachment.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    backToColorAttachment.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    backToColorAttachment.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    backToColorAttachment.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    backToColorAttachment.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    backToColorAttachment.image = offscreenImage;
    backToColorAttachment.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    backToColorAttachment.subresourceRange.baseMipLevel = 0;
    backToColorAttachment.subresourceRange.levelCount = 1;
    backToColorAttachment.subresourceRange.baseArrayLayer = 0;
    backToColorAttachment.subresourceRange.layerCount = 1;
    backToColorAttachment.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    backToColorAttachment.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &backToColorAttachment);

    endSingleTimeCommands(commandBuffer);

    void *mappedData = nullptr;
    if (vmaMapMemory(MemorySystem::allocator(),
                     offscreenReadbackAllocation.vmaAllocation,
                     &mappedData) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to map offscreen readback buffer.");
    }

    std::memcpy(outPixels.data(), mappedData, static_cast<size_t>(byteSize));
    vmaUnmapMemory(MemorySystem::allocator(),
                   offscreenReadbackAllocation.vmaAllocation);
}

void SplatCoreApp::recordCommandBuffer(VkCommandBuffer cmdBuffer, uint32_t imageIndex)
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(cmdBuffer, &beginInfo) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to begin command buffer.");
    }

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = swapChainExtent;

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {{0.05f, 0.05f, 0.1f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(cmdBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(swapChainExtent.width);
    viewport.height = static_cast<float>(swapChainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapChainExtent;
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    // Bind point cloud vertex buffer
    const VkBuffer vertexBuffers[] = {pointVertexBuffer};
    const VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(cmdBuffer, 0, 1, vertexBuffers, offsets);

    // Bind descriptor set (UBO with camera MVP)
    vkCmdBindDescriptorSets(cmdBuffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout,
                            0, 1, &descriptorSets[currentFrame],
                            0, nullptr);

    // Draw all points
    vkCmdDraw(cmdBuffer, pointCount, 1, 0, 0);

    vkCmdEndRenderPass(cmdBuffer);

    VkImageMemoryBarrier swapchainToTransferSrc{};
    swapchainToTransferSrc.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    swapchainToTransferSrc.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    swapchainToTransferSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    swapchainToTransferSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapchainToTransferSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapchainToTransferSrc.image = swapChainImages[imageIndex];
    swapchainToTransferSrc.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    swapchainToTransferSrc.subresourceRange.baseMipLevel = 0;
    swapchainToTransferSrc.subresourceRange.levelCount = 1;
    swapchainToTransferSrc.subresourceRange.baseArrayLayer = 0;
    swapchainToTransferSrc.subresourceRange.layerCount = 1;
    swapchainToTransferSrc.srcAccessMask = 0;
    swapchainToTransferSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(cmdBuffer,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &swapchainToTransferSrc);

    VkImageCopy copyRegion{};
    copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.srcSubresource.mipLevel = 0;
    copyRegion.srcSubresource.baseArrayLayer = 0;
    copyRegion.srcSubresource.layerCount = 1;
    copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.dstSubresource.mipLevel = 0;
    copyRegion.dstSubresource.baseArrayLayer = 0;
    copyRegion.dstSubresource.layerCount = 1;
    copyRegion.extent = {
        swapChainExtent.width,
        swapChainExtent.height,
        1};

    vkCmdCopyImage(cmdBuffer,
                   swapChainImages[imageIndex],
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   offscreenImage,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1,
                   &copyRegion);

    VkImageMemoryBarrier swapchainBackToPresent{};
    swapchainBackToPresent.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    swapchainBackToPresent.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    swapchainBackToPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    swapchainBackToPresent.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapchainBackToPresent.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapchainBackToPresent.image = swapChainImages[imageIndex];
    swapchainBackToPresent.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    swapchainBackToPresent.subresourceRange.baseMipLevel = 0;
    swapchainBackToPresent.subresourceRange.levelCount = 1;
    swapchainBackToPresent.subresourceRange.baseArrayLayer = 0;
    swapchainBackToPresent.subresourceRange.layerCount = 1;
    swapchainBackToPresent.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    swapchainBackToPresent.dstAccessMask = 0;

    vkCmdPipelineBarrier(cmdBuffer,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &swapchainBackToPresent);

    if (vkEndCommandBuffer(cmdBuffer) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to record command buffer.");
    }
}

std::vector<char> SplatCoreApp::readSpvFile(const std::string &filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open shader file: " + filename);
    }
    const size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
    return buffer;
}

VkShaderModule SplatCoreApp::createShaderModule(const std::vector<char> &code)
{
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create shader module.");
    }
    return shaderModule;
}

void SplatCoreApp::createGraphicsPipeline()
{
    // SHADER_DIR is injected by CMake as an absolute path string.
    const std::string shaderDir = SHADER_DIR;
    const std::vector<char> vertCode = readSpvFile(shaderDir + "point.vert.spv");
    const std::vector<char> fragCode = readSpvFile(shaderDir + "point.frag.spv");

    VkShaderModule vertShaderModule = createShaderModule(vertCode);
    VkShaderModule fragShaderModule = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo vertStageInfo{};
    vertStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStageInfo.module = vertShaderModule;
    vertStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragStageInfo{};
    fragStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStageInfo.module = fragShaderModule;
    fragStageInfo.pName = "main";

    const VkPipelineShaderStageCreateInfo shaderStages[] = {
        vertStageInfo, fragStageInfo};

    // Vertex input: one binding, two attributes (position + color).
    const auto bindingDescription = Vertex::getBindingDescription();
    const auto attributeDescriptions = Vertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // Dynamic viewport and scissor — set per-frame in recordCommandBuffer().
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = nullptr; // set dynamically
    viewportState.scissorCount = 1;
    viewportState.pScissors = nullptr; // set dynamically

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                               &pipelineLayout) != VK_SUCCESS)
    {
        // Clean up shader modules before propagating the error.
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
        throw std::runtime_error("Failed to create pipeline layout.");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    const VkDynamicState dynamicStates[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicStateInfo{};
    dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicStateInfo.dynamicStateCount = 2;
    dynamicStateInfo.pDynamicStates = dynamicStates;

    pipelineInfo.pDynamicState = &dynamicStateInfo;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;

    const VkResult result = vkCreateGraphicsPipelines(
        device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline);

    // Shader modules are never needed after pipeline creation — destroy immediately.
    // This satisfies the Validation Layer requirement.
    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);

    if (result != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create graphics pipeline.");
    }
}

// ── Descriptor Set Layout ────────────────────────────────────────────────
void SplatCoreApp::createDescriptorSetLayout()
{
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    uboLayoutBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &uboLayoutBinding;

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                                    &descriptorSetLayout) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create descriptor set layout.");
    }
}

// ── Memory utility ───────────────────────────────────────────────────────
uint32_t SplatCoreApp::findMemoryType(uint32_t typeFilter,
                                      VkMemoryPropertyFlags properties) const
{
    VkPhysicalDeviceMemoryProperties memProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i)
    {
        if ((typeFilter & (1U << i)) != 0U &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type.");
}

void SplatCoreApp::destroyBufferAllocation(VkBuffer &buffer,
                                           Allocation &allocation)
{
    if (allocation.vmaAllocation != nullptr)
    {
        MemorySystem::free(allocation);
    }
    buffer = VK_NULL_HANDLE;
}

// ── Generic buffer creator ───────────────────────────────────────────────
void SplatCoreApp::createBuffer(VkDeviceSize size,
                                VkBufferUsageFlags usage,
                                VkMemoryPropertyFlags properties,
                                VkBuffer &buffer,
                                Allocation &allocation,
                                MemoryRegion region,
                                std::string_view allocationName)
{
    buffer = VK_NULL_HANDLE;
    allocation = {};

    AllocationDesc desc{};
    desc.region = region;
    desc.bufferUsage = usage;
    if ((properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0)
    {
        desc.vmaUsage = (usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) != 0
                            ? VMA_MEMORY_USAGE_CPU_ONLY
                            : VMA_MEMORY_USAGE_CPU_TO_GPU;
    }
    else
    {
        desc.vmaUsage = VMA_MEMORY_USAGE_GPU_ONLY;
    }
    desc.size = size;
    desc.allocationName = allocationName;
    desc.imageInfo = nullptr;

    allocation = MemorySystem::allocate(desc);
    buffer = allocation.buffer;
}

// ── Single-time command helpers ──────────────────────────────────────────
VkCommandBuffer SplatCoreApp::beginSingleTimeCommands()
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate single-time command buffer.");
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
    {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        throw std::runtime_error("Failed to begin single-time command buffer.");
    }

    return commandBuffer;
}

void SplatCoreApp::endSingleTimeCommands(VkCommandBuffer commandBuffer)
{
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
    {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        throw std::runtime_error("Failed to end single-time command buffer.");
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS)
    {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        throw std::runtime_error("Failed to submit single-time command buffer.");
    }
    if (vkQueueWaitIdle(graphicsQueue) != VK_SUCCESS)
    {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        throw std::runtime_error("Failed waiting for single-time command buffer.");
    }

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

// ── Buffer copy ──────────────────────────────────────────────────────────
void SplatCoreApp::copyBuffer(VkBuffer srcBuffer,
                              VkBuffer dstBuffer,
                              VkDeviceSize size)
{
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    endSingleTimeCommands(commandBuffer);
}

// ── Image creation helper ─────────────────────────────────────────────────
void SplatCoreApp::destroyImageAllocation(VkImage &image,
                                          Allocation &allocation)
{
    if (allocation.vmaAllocation != nullptr)
    {
        MemorySystem::free(allocation);
    }
    image = VK_NULL_HANDLE;
}

void SplatCoreApp::createImage(uint32_t width, uint32_t height,
                               VkFormat format, VkImageTiling tiling,
                               VkImageUsageFlags usage,
                               VkMemoryPropertyFlags properties,
                               VkImage &image,
                               Allocation &allocation,
                               MemoryRegion region,
                               std::string_view allocationName)
{
    image = VK_NULL_HANDLE;
    allocation = {};

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    AllocationDesc desc{};
    desc.region = region;
    desc.bufferUsage = 0;
    desc.vmaUsage = (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0
                        ? VMA_MEMORY_USAGE_AUTO_PREFER_HOST
                        : VMA_MEMORY_USAGE_GPU_ONLY;
    desc.size = 0;
    desc.allocationName = allocationName;
    desc.imageInfo = &imageInfo;

    allocation = MemorySystem::allocate(desc);
    image = allocation.image;
    if (image == VK_NULL_HANDLE)
    {
        throw std::runtime_error("MemorySystem::allocate returned null VkImage.");
    }
}

VkImageView SplatCoreApp::createImageView(VkImage image, VkFormat format,
                                          VkImageAspectFlags aspectFlags)
{
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView = VK_NULL_HANDLE;
    if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS)
        throw std::runtime_error("Failed to create image view.");
    return imageView;
}

// ── Depth format selection ────────────────────────────────────────────────
VkFormat SplatCoreApp::findSupportedFormat(
    const std::vector<VkFormat> &candidates,
    VkImageTiling tiling,
    VkFormatFeatureFlags features) const
{
    for (VkFormat fmt : candidates)
    {
        VkFormatProperties props{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, fmt, &props);
        if (tiling == VK_IMAGE_TILING_LINEAR &&
            (props.linearTilingFeatures & features) == features)
            return fmt;
        if (tiling == VK_IMAGE_TILING_OPTIMAL &&
            (props.optimalTilingFeatures & features) == features)
            return fmt;
    }
    throw std::runtime_error("Failed to find supported depth format.");
}

VkFormat SplatCoreApp::findDepthFormat() const
{
    return findSupportedFormat(
        {VK_FORMAT_D32_SFLOAT,
         VK_FORMAT_D32_SFLOAT_S8_UINT,
         VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

// ── Depth resource creation ───────────────────────────────────────────────
void SplatCoreApp::createDepthResources()
{
    const VkFormat depthFormat = findDepthFormat();
    createImage(swapChainExtent.width, swapChainExtent.height,
                depthFormat,
                VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                depthImage,
                depthImageAllocation,
                MemoryRegion::STATIC,
                "GaussianRenderer::DepthImage");
    try
    {
        depthImageView = createImageView(depthImage, depthFormat,
                                         VK_IMAGE_ASPECT_DEPTH_BIT);
    }
    catch (...)
    {
        destroyImageAllocation(depthImage, depthImageAllocation);
        throw;
    }
}

// ── Binary PLY loader ─────────────────────────────────────────────────────
void SplatCoreApp::loadPointCloud()
{
    if (plyFilePath.empty())
        throw std::runtime_error("PLY file path not set.");

    std::cout << "[PLY] Loading: " << plyFilePath << std::endl;

    // Reload-safe: release previous point cloud ownership before reallocation.
    destroyBufferAllocation(pointVertexBuffer, pointVertexBufferAllocation);
    pointCount = 0;

    std::ifstream file(plyFilePath, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Failed to open PLY file: " + plyFilePath);

    enum class PlyFormat
    {
        ASCII,
        BinaryLittleEndian,
        BinaryBigEndian
    };
    struct PlyProperty
    {
        std::string name;
        std::string type;
        bool isList = false;
        std::string listCountType;
        std::string listValueType;
        size_t offset = 0;
        size_t size = 0;
    };
    struct PlyElement
    {
        std::string name;
        uint32_t count = 0;
        std::vector<PlyProperty> properties;
        size_t stride = 0;
    };

    auto scalarTypeSize = [](const std::string &type) -> size_t
    {
        if (type == "char" || type == "uchar" ||
            type == "int8" || type == "uint8")
            return 1;
        if (type == "short" || type == "ushort" ||
            type == "int16" || type == "uint16")
            return 2;
        if (type == "int" || type == "uint" ||
            type == "int32" || type == "uint32" ||
            type == "float" || type == "float32")
            return 4;
        if (type == "double" || type == "float64")
            return 8;
        throw std::runtime_error("Unsupported PLY scalar type: " + type);
    };
    auto isIntegerType = [](const std::string &type) -> bool
    {
        return type == "char" || type == "uchar" ||
               type == "int8" || type == "uint8" ||
               type == "short" || type == "ushort" ||
               type == "int16" || type == "uint16" ||
               type == "int" || type == "uint" ||
               type == "int32" || type == "uint32";
    };
    auto integerTypeMax = [](const std::string &type) -> double
    {
        if (type == "uchar" || type == "uint8")
            return 255.0;
        if (type == "char" || type == "int8")
            return 127.0;
        if (type == "ushort" || type == "uint16")
            return 65535.0;
        if (type == "short" || type == "int16")
            return 32767.0;
        if (type == "uint" || type == "uint32")
            return static_cast<double>(std::numeric_limits<uint32_t>::max());
        if (type == "int" || type == "int32")
            return static_cast<double>(std::numeric_limits<int32_t>::max());
        return 1.0;
    };
    auto hostIsLittleEndian = []() -> bool
    {
        const uint16_t value = 1;
        return *reinterpret_cast<const uint8_t *>(&value) == 1;
    };
    auto fileNeedsByteSwap = [&](PlyFormat format) -> bool
    {
        if (format == PlyFormat::ASCII)
            return false;
        return hostIsLittleEndian()
                   ? (format == PlyFormat::BinaryBigEndian)
                   : (format == PlyFormat::BinaryLittleEndian);
    };
    auto scalarFromBytes = [&](const char *src,
                               const std::string &type,
                               PlyFormat format) -> double
    {
        std::array<uint8_t, 8> raw{};
        const size_t size = scalarTypeSize(type);
        std::memcpy(raw.data(), src, size);
        if (fileNeedsByteSwap(format) && size > 1)
            std::reverse(raw.begin(), raw.begin() + static_cast<std::ptrdiff_t>(size));

        if (type == "char" || type == "int8")
        {
            int8_t value = 0;
            std::memcpy(&value, raw.data(), sizeof(value));
            return static_cast<double>(value);
        }
        if (type == "uchar" || type == "uint8")
        {
            uint8_t value = 0;
            std::memcpy(&value, raw.data(), sizeof(value));
            return static_cast<double>(value);
        }
        if (type == "short" || type == "int16")
        {
            int16_t value = 0;
            std::memcpy(&value, raw.data(), sizeof(value));
            return static_cast<double>(value);
        }
        if (type == "ushort" || type == "uint16")
        {
            uint16_t value = 0;
            std::memcpy(&value, raw.data(), sizeof(value));
            return static_cast<double>(value);
        }
        if (type == "int" || type == "int32")
        {
            int32_t value = 0;
            std::memcpy(&value, raw.data(), sizeof(value));
            return static_cast<double>(value);
        }
        if (type == "uint" || type == "uint32")
        {
            uint32_t value = 0;
            std::memcpy(&value, raw.data(), sizeof(value));
            return static_cast<double>(value);
        }
        if (type == "float" || type == "float32")
        {
            float value = 0.0f;
            std::memcpy(&value, raw.data(), sizeof(value));
            return static_cast<double>(value);
        }
        if (type == "double" || type == "float64")
        {
            double value = 0.0;
            std::memcpy(&value, raw.data(), sizeof(value));
            return value;
        }

        throw std::runtime_error("Unsupported PLY scalar type: " + type);
    };
    auto scalarFromToken = [&](const std::string &token,
                               const std::string &type) -> double
    {
        if (type == "char" || type == "int8" ||
            type == "short" || type == "int16" ||
            type == "int" || type == "int32")
        {
            return static_cast<double>(std::stoll(token));
        }
        if (type == "uchar" || type == "uint8" ||
            type == "ushort" || type == "uint16" ||
            type == "uint" || type == "uint32")
        {
            return static_cast<double>(std::stoull(token));
        }
        return std::stod(token);
    };
    auto normalizeColor = [&](double value, const std::string &type) -> float
    {
        if (isIntegerType(type))
            value /= integerTypeMax(type);
        return std::clamp(static_cast<float>(value), 0.0f, 1.0f);
    };

    std::vector<PlyElement> elements;
    PlyElement *currentElement = nullptr;
    PlyFormat format = PlyFormat::ASCII;
    bool formatSeen = false;
    uint32_t vertexCount = 0;

    std::string line;
    while (std::getline(file, line))
    {
        // Trim Windows CR
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line == "end_header")
            break;

        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "format")
        {
            std::string fmt;
            iss >> fmt;
            if (fmt == "ascii")
                format = PlyFormat::ASCII;
            else if (fmt == "binary_little_endian")
                format = PlyFormat::BinaryLittleEndian;
            else if (fmt == "binary_big_endian")
                format = PlyFormat::BinaryBigEndian;
            else
                throw std::runtime_error("Unsupported PLY format: " + fmt);
            formatSeen = true;
        }
        else if (token == "element")
        {
            std::string elem;
            uint32_t count = 0;
            iss >> elem >> count;
            elements.push_back({elem, count});
            currentElement = &elements.back();
            if (elem == "vertex")
                vertexCount = count;
        }
        else if (token == "property")
        {
            if (currentElement == nullptr)
                continue;

            std::string typeOrList;
            iss >> typeOrList;

            PlyProperty prop{};
            if (typeOrList == "list")
            {
                prop.isList = true;
                iss >> prop.listCountType >> prop.listValueType >> prop.name;
            }
            else
            {
                prop.type = typeOrList;
                iss >> prop.name;
                prop.size = scalarTypeSize(prop.type);
                prop.offset = currentElement->stride;
                currentElement->stride += prop.size;
            }

            currentElement->properties.push_back(prop);
        }
    }

    if (!formatSeen)
        throw std::runtime_error("PLY header missing format.");
    if (vertexCount == 0)
        throw std::runtime_error("PLY has no vertices.");

    const PlyElement *vertexElement = nullptr;
    for (const PlyElement &element : elements)
    {
        if (element.name == "vertex")
        {
            vertexElement = &element;
            break;
        }
    }
    if (vertexElement == nullptr)
        throw std::runtime_error("PLY missing vertex element.");

    const PlyProperty *propX = nullptr;
    const PlyProperty *propY = nullptr;
    const PlyProperty *propZ = nullptr;
    const PlyProperty *propR = nullptr;
    const PlyProperty *propG = nullptr;
    const PlyProperty *propB = nullptr;
    const PlyProperty *propShR = nullptr;
    const PlyProperty *propShG = nullptr;
    const PlyProperty *propShB = nullptr;
    for (const PlyProperty &property : vertexElement->properties)
    {
        if (property.isList)
            throw std::runtime_error("PLY vertex list properties are not supported.");

        if (property.name == "x")
            propX = &property;
        else if (property.name == "y")
            propY = &property;
        else if (property.name == "z")
            propZ = &property;
        else if (property.name == "red")
            propR = &property;
        else if (property.name == "green")
            propG = &property;
        else if (property.name == "blue")
            propB = &property;
        else if (property.name == "f_dc_0")
            propShR = &property;
        else if (property.name == "f_dc_1")
            propShG = &property;
        else if (property.name == "f_dc_2")
            propShB = &property;
    }
    if (propX == nullptr || propY == nullptr || propZ == nullptr)
        throw std::runtime_error("PLY missing x/y/z properties.");

    const bool hasRgb = (propR != nullptr && propG != nullptr && propB != nullptr);
    const bool hasShDc = (propShR != nullptr || propShG != nullptr || propShB != nullptr);
    constexpr float kSH0 = 0.2820947918f;

    std::vector<Vertex> vertices(vertexCount);
    if (format == PlyFormat::ASCII)
    {
        for (uint32_t i = 0; i < vertexCount; ++i)
        {
            if (!std::getline(file, line))
                throw std::runtime_error("PLY ASCII read failed (truncated?).");
            if (!line.empty() && line.back() == '\r')
                line.pop_back();

            std::istringstream row(line);
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            float r = 0.5f;
            float g = 0.5f;
            float b = 0.5f;

            for (const PlyProperty &property : vertexElement->properties)
            {
                std::string tokenValue;
                if (!(row >> tokenValue))
                    throw std::runtime_error("PLY ASCII vertex row is incomplete.");

                const double value = scalarFromToken(tokenValue, property.type);
                if (property.name == "x")
                    x = value;
                else if (property.name == "y")
                    y = value;
                else if (property.name == "z")
                    z = value;
                else if (hasRgb && property.name == "red")
                    r = normalizeColor(value, property.type);
                else if (hasRgb && property.name == "green")
                    g = normalizeColor(value, property.type);
                else if (hasRgb && property.name == "blue")
                    b = normalizeColor(value, property.type);
                else if (!hasRgb && property.name == "f_dc_0")
                    r = std::clamp(0.5f + kSH0 * static_cast<float>(value), 0.0f, 1.0f);
                else if (!hasRgb && property.name == "f_dc_1")
                    g = std::clamp(0.5f + kSH0 * static_cast<float>(value), 0.0f, 1.0f);
                else if (!hasRgb && property.name == "f_dc_2")
                    b = std::clamp(0.5f + kSH0 * static_cast<float>(value), 0.0f, 1.0f);
            }

            vertices[i].pos = {static_cast<float>(x),
                               static_cast<float>(y),
                               static_cast<float>(z)};
            vertices[i].color = {r, g, b};
        }
    }
    else
    {
        const size_t totalBytes = static_cast<size_t>(vertexCount) * vertexElement->stride;
        std::vector<char> rawData(totalBytes);
        file.read(rawData.data(), static_cast<std::streamsize>(totalBytes));
        if (!file)
            throw std::runtime_error("PLY binary read failed (truncated?).");

        std::cout << "[PLY] Parsed " << vertexCount << " vertices ("
                  << (totalBytes >> 20) << " MB raw)" << std::endl;

        for (uint32_t i = 0; i < vertexCount; ++i)
        {
            const char *base = rawData.data() +
                               static_cast<size_t>(i) * vertexElement->stride;

            const double x = scalarFromBytes(base + propX->offset, propX->type, format);
            const double y = scalarFromBytes(base + propY->offset, propY->type, format);
            const double z = scalarFromBytes(base + propZ->offset, propZ->type, format);

            float r = 0.5f;
            float g = 0.5f;
            float b = 0.5f;
            if (hasRgb)
            {
                r = normalizeColor(scalarFromBytes(base + propR->offset, propR->type, format), propR->type);
                g = normalizeColor(scalarFromBytes(base + propG->offset, propG->type, format), propG->type);
                b = normalizeColor(scalarFromBytes(base + propB->offset, propB->type, format), propB->type);
            }
            else if (hasShDc)
            {
                if (propShR != nullptr)
                    r = std::clamp(0.5f + kSH0 * static_cast<float>(scalarFromBytes(base + propShR->offset, propShR->type, format)), 0.0f, 1.0f);
                if (propShG != nullptr)
                    g = std::clamp(0.5f + kSH0 * static_cast<float>(scalarFromBytes(base + propShG->offset, propShG->type, format)), 0.0f, 1.0f);
                if (propShB != nullptr)
                    b = std::clamp(0.5f + kSH0 * static_cast<float>(scalarFromBytes(base + propShB->offset, propShB->type, format)), 0.0f, 1.0f);
            }

            vertices[i].pos = {static_cast<float>(x),
                               static_cast<float>(y),
                               static_cast<float>(z)};
            vertices[i].color = {r, g, b};
        }
    }
    file.close();

    if (format == PlyFormat::ASCII)
    {
        std::cout << "[PLY] Parsed " << vertexCount << " vertices (ASCII)"
                  << std::endl;
    }
    pointCount = vertexCount;

    // ── Upload via staging buffer ─────────────────────────────────────────
    const VkDeviceSize bufferSize = sizeof(Vertex) * vertices.size();

    VkBuffer stagingBuf = VK_NULL_HANDLE;
    Allocation stagingAllocation{};
    try
    {
        createBuffer(bufferSize,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuf,
                     stagingAllocation,
                     MemoryRegion::STAGING,
                     "TextureUploader::PointCloudStagingBuffer");

        void *data = nullptr;
        if (vmaMapMemory(MemorySystem::allocator(),
                         stagingAllocation.vmaAllocation,
                         &data) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to map staging buffer memory.");
        }
        std::memcpy(data, vertices.data(), static_cast<size_t>(bufferSize));
        vmaUnmapMemory(MemorySystem::allocator(),
                       stagingAllocation.vmaAllocation);

        createBuffer(bufferSize,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     pointVertexBuffer,
                     pointVertexBufferAllocation,
                     MemoryRegion::STATIC,
                     "GaussianRenderer::SplatVertexBuffer");

        copyBuffer(stagingBuf, pointVertexBuffer, bufferSize);
    }
    catch (...)
    {
        destroyBufferAllocation(stagingBuf, stagingAllocation);
        destroyBufferAllocation(pointVertexBuffer, pointVertexBufferAllocation);
        pointCount = 0;
        throw;
    }

    destroyBufferAllocation(stagingBuf, stagingAllocation);

    std::cout << "[PLY] GPU upload complete — " << pointCount << " points, "
              << (bufferSize >> 20) << " MB VRAM." << std::endl;
}

// ── Input callbacks ───────────────────────────────────────────────────────
void SplatCoreApp::cursorPosCallback(GLFWwindow *window,
                                     double xpos, double ypos)
{
    auto *app = reinterpret_cast<SplatCoreApp *>(
        glfwGetWindowUserPointer(window));
    if (app == nullptr || !app->mouseCaptured)
        return;

    if (app->firstMouse)
    {
        app->lastMouseX = xpos;
        app->lastMouseY = ypos;
        app->firstMouse = false;
        return;
    }

    const float dx = static_cast<float>(xpos - app->lastMouseX);
    const float dy = static_cast<float>(app->lastMouseY - ypos); // Y inverted
    app->lastMouseX = xpos;
    app->lastMouseY = ypos;
    app->camera.processMouse(dx, dy);
}

void SplatCoreApp::mouseButtonCallback(GLFWwindow *window,
                                       int button, int action, int /*mods*/)
{
    auto *app = reinterpret_cast<SplatCoreApp *>(
        glfwGetWindowUserPointer(window));
    if (app == nullptr)
        return;

    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        app->mouseCaptured = true;
        app->firstMouse = true;
    }
    else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
    {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        app->mouseCaptured = false;
    }
}

void SplatCoreApp::scrollCallback(GLFWwindow *window,
                                  double /*xoffset*/, double yoffset)
{
    auto *app = reinterpret_cast<SplatCoreApp *>(
        glfwGetWindowUserPointer(window));
    if (app == nullptr)
        return;
    // Scroll up = speed up, scroll down = slow down
    app->camera.speed = std::clamp(
        app->camera.speed + static_cast<float>(yoffset) * 0.5f,
        0.1f, 200.0f);
}

// ── Uniform buffers (one per frame-in-flight, persistently mapped) ────────
void SplatCoreApp::createUniformBuffers()
{
    const VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT, VK_NULL_HANDLE);
    uniformBufferAllocations.resize(MAX_FRAMES_IN_FLIGHT, {});
    uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT, nullptr);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        createBuffer(bufferSize,
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     uniformBuffers[i],
                     uniformBufferAllocations[i],
                     MemoryRegion::STATIC,
                     "GaussianRenderer::PerFrameUniformBuffer");

        // Persistent map — never unmap until destroy.
        if (vmaMapMemory(MemorySystem::allocator(),
                         uniformBufferAllocations[i].vmaAllocation,
                         &uniformBuffersMapped[i]) != VK_SUCCESS)
        {
            destroyBufferAllocation(uniformBuffers[i], uniformBufferAllocations[i]);
            throw std::runtime_error("Failed to map uniform buffer memory.");
        }
    }
}

// ── Descriptor pool ──────────────────────────────────────────────────────
void SplatCoreApp::createDescriptorPool()
{
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr,
                               &descriptorPool) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create descriptor pool.");
    }
}

// ── Descriptor sets ──────────────────────────────────────────────────────
void SplatCoreApp::createDescriptorSets()
{
    std::vector<VkDescriptorSetLayout> layouts(
        MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    if (vkAllocateDescriptorSets(device, &allocInfo,
                                 descriptorSets.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate descriptor sets.");
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);

        VkWriteDescriptorSet descriptorWrite{};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = descriptorSets[i];
        descriptorWrite.dstBinding = 0;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
    }
}

// ── MVP matrix update (called once per frame) ────────────────────────────
void SplatCoreApp::updateUniformBuffer(uint32_t frameSlot)
{
    UniformBufferObject ubo{};

    // Model: identity — point cloud coordinates are already in world space.
    ubo.model = glm::mat4(1.0f);

    // View: FPS camera.
    ubo.view = camera.getView();

    // Projection: 60° FOV, current aspect ratio, wide depth range for large scenes.
    ubo.proj = glm::perspective(
        glm::radians(60.0f),
        static_cast<float>(swapChainExtent.width) /
            static_cast<float>(swapChainExtent.height),
        0.01f, 1000.0f);

    // Vulkan Y-flip.
    ubo.proj[1][1] *= -1.0f;

    std::memcpy(uniformBuffersMapped[frameSlot], &ubo, sizeof(ubo));
}

bool SplatCoreApp::checkValidationLayerSupport() const
{
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char *layerName : validationLayers)
    {
        bool layerFound = false;
        for (const VkLayerProperties &layerProperties : availableLayers)
        {
            if (std::strcmp(layerName, layerProperties.layerName) == 0)
            {
                layerFound = true;
                break;
            }
        }

        if (!layerFound)
        {
            return false;
        }
    }

    return true;
}

std::vector<const char *> SplatCoreApp::getRequiredExtensions() const
{
    uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    if (glfwExtensions == nullptr)
    {
        throw std::runtime_error("GLFW did not return required Vulkan instance extensions.");
    }

    std::vector<const char *> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
#ifdef VK_ENABLE_VALIDATION_LAYERS
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif
    return extensions;
}

bool SplatCoreApp::isDeviceSuitable(VkPhysicalDevice candidateDevice) const
{
    const QueueFamilyIndices indices = findQueueFamilies(candidateDevice);
    const bool extensionsSupported = checkDeviceExtensionSupport(candidateDevice);

    bool swapChainAdequate = false;
    if (extensionsSupported)
    {
        const SwapChainSupportDetails swapChainSupport = querySwapChainSupport(candidateDevice);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    return indices.isComplete() && extensionsSupported && swapChainAdequate;
}

bool SplatCoreApp::checkDeviceExtensionSupport(VkPhysicalDevice candidateDevice) const
{
    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(candidateDevice, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(candidateDevice, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
    for (const VkExtensionProperties &extension : availableExtensions)
    {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

SplatCoreApp::QueueFamilyIndices SplatCoreApp::findQueueFamilies(VkPhysicalDevice candidateDevice) const
{
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(candidateDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(candidateDevice, &queueFamilyCount, queueFamilies.data());

    for (uint32_t i = 0; i < queueFamilyCount; ++i)
    {
        if ((queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0U)
        {
            indices.graphicsFamily = i;
        }

        VkBool32 presentSupport = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(candidateDevice, i, surface, &presentSupport);
        if (presentSupport == VK_TRUE)
        {
            indices.presentFamily = i;
        }

        if (indices.isComplete())
        {
            break;
        }
    }

    return indices;
}

SplatCoreApp::SwapChainSupportDetails SplatCoreApp::querySwapChainSupport(VkPhysicalDevice candidateDevice) const
{
    SwapChainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(candidateDevice, surface, &details.capabilities);

    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(candidateDevice, surface, &formatCount, nullptr);
    if (formatCount != 0)
    {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(candidateDevice, surface, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(candidateDevice, surface, &presentModeCount, nullptr);
    if (presentModeCount != 0)
    {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(candidateDevice, surface, &presentModeCount, details.presentModes.data());
    }

    return details;
}

VkSurfaceFormatKHR SplatCoreApp::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats)
{
    for (const VkSurfaceFormatKHR &availableFormat : availableFormats)
    {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
            availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            return availableFormat;
        }
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM &&
            availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

VkPresentModeKHR SplatCoreApp::chooseSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes)
{
    for (const VkPresentModeKHR mode : availablePresentModes)
    {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            return mode; // low-latency triple-buffer if available
        }
    }
    // VK_PRESENT_MODE_FIFO_KHR is guaranteed by the Vulkan spec.
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D SplatCoreApp::chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities) const
{
    if (capabilities.currentExtent.width != UINT32_MAX)
    {
        return capabilities.currentExtent;
    }

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);

    VkExtent2D actualExtent = {
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height)};

    actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

    return actualExtent;
}

void SplatCoreApp::failIfValidationIssueDetected() const
{
    if (validationErrorDetected)
    {
        throw std::runtime_error(validationErrorMessage.empty() ? "Validation error detected." : validationErrorMessage);
    }
}

void SplatCoreApp::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo)
{
    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
}

VKAPI_ATTR VkBool32 VKAPI_CALL SplatCoreApp::debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
    void *pUserData)
{
    const char *message = (pCallbackData != nullptr && pCallbackData->pMessage != nullptr) ? pCallbackData->pMessage : "No message";

    if ((messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0U)
    {
        std::cerr << "[Validation][ERROR][Type:" << messageType << "] " << message << std::endl;

        SplatCoreApp *app = reinterpret_cast<SplatCoreApp *>(pUserData);
        if (app != nullptr)
        {
            app->validationErrorDetected = true;
            app->validationErrorMessage = std::string("Validation Error: ") + message;
        }
        return VK_FALSE;
    }

    if ((messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0U)
    {
        // Ignore duplicate implicit-layer noise emitted by some launcher overlays.
        if (std::strstr(message, "Removing layer VK_LAYER_EOS_Overlay") != nullptr)
        {
            return VK_FALSE;
        }
        // Ignore BestPractices small-allocation advisory for tiny demo buffers.
        if (std::strstr(message, "should be sub-allocated from larger memory blocks") != nullptr)
        {
            return VK_FALSE;
        }
        std::cout << "[Validation][WARNING][Type:" << messageType << "] " << message << std::endl;
        return VK_FALSE;
    }

    if ((messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) != 0U)
    {
        std::cout << "[Validation][INFO][Type:" << messageType << "] " << message << std::endl;
        return VK_FALSE;
    }

    std::cout << "[Validation][VERBOSE][Type:" << messageType << "] " << message << std::endl;
    return VK_FALSE;
}

VkResult SplatCoreApp::createDebugUtilsMessengerEXT(
    VkInstance vkInstance,
    const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugUtilsMessengerEXT *pDebugMessenger)
{
    const auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(vkInstance, "vkCreateDebugUtilsMessengerEXT"));
    if (func != nullptr)
    {
        return func(vkInstance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void SplatCoreApp::destroyDebugUtilsMessengerEXT(
    VkInstance vkInstance,
    VkDebugUtilsMessengerEXT vkDebugMessenger,
    const VkAllocationCallbacks *pAllocator)
{
    const auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(vkInstance, "vkDestroyDebugUtilsMessengerEXT"));
    if (func != nullptr)
    {
        func(vkInstance, vkDebugMessenger, pAllocator);
    }
}

#ifndef SPLATCORE_NO_ENTRYPOINT
int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "[Fatal] Usage: SplatCore_SilasHan.exe <path_to_file.ply>"
                  << std::endl;
        return EXIT_FAILURE;
    }

    try
    {
        SplatCoreApp app;
        app.setPlyPath(argv[1]);
        app.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Fatal] " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
#endif
