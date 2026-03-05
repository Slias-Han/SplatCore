#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <atomic>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

class SplatCoreApp
{
public:
    ~SplatCoreApp();
    void run();

private:
    static constexpr uint32_t kWidth = 800;
    static constexpr uint32_t kHeight = 600;
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
    static constexpr const char *kWindowTitle = "SplatCore v0.3";

#ifdef VK_ENABLE_VALIDATION_LAYERS
    const std::vector<const char *> validationLayers = {
        "VK_LAYER_KHRONOS_validation"};
#else
    const std::vector<const char *> validationLayers;   // empty in Release
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
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline    = VK_NULL_HANDLE;
    // Stores the currentFrame index (0..MAX_FRAMES_IN_FLIGHT-1)
    // that last submitted to this swapchain image, or kNoFrame.
    static constexpr uint32_t kNoFrame = UINT32_MAX;
    std::vector<uint32_t> imagesInFlight;
    uint32_t currentFrame = 0;
    bool framebufferResized = false;
    double lastFpsTime     = 0.0;
    uint32_t frameCount    = 0;

    std::atomic<bool> validationErrorDetected{false};
    std::string validationErrorMessage;

    void initWindow();
    void initVulkan();
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
    void cleanupSwapChain();
    void recreateSwapChain();
    void createGraphicsPipeline();
    static void framebufferResizeCallback(GLFWwindow *window,
                                          int width, int height);
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
    initWindow();
    initVulkan();
    mainLoop();
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
    lastFpsTime = glfwGetTime();
}

void SplatCoreApp::initVulkan()
{
    createInstance();
#ifdef VK_ENABLE_VALIDATION_LAYERS
    setupDebugMessenger();
#endif
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createGraphicsPipeline();
    createFramebuffers();
    createCommandPool();
    createCommandBuffer();
    createSyncObjects();
    failIfValidationIssueDetected();
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

    // Rebuild swapchain-dependent resources.
    createSwapChain();
    createImageViews();
    createFramebuffers();

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

    cleanupSwapChain();   // destroys framebuffers, imageViews, swapchain

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

    // 5) Record command buffer.
    VkCommandBuffer currentCommandBuffer = commandBuffers[currentFrame];
    if (vkResetCommandBuffer(currentCommandBuffer, 0) != VK_SUCCESS)
    {
        throw std::runtime_error("vkResetCommandBuffer failed.");
    }
    recordCommandBuffer(currentCommandBuffer, imageIndex);

    // 6) Submit.
    const VkSemaphore waitSemaphores[]   = {imageAvailableSemaphores[currentFrame]};
    const VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    const VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[imageIndex]};

    VkSubmitInfo submitInfo{};
    submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount   = 1;
    submitInfo.pWaitSemaphores      = waitSemaphores;
    submitInfo.pWaitDstStageMask    = waitStages;
    submitInfo.commandBufferCount   = 1;
    submitInfo.pCommandBuffers      = &currentCommandBuffer;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores    = signalSemaphores;

    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo,
                      inFlightFences[currentFrame]) != VK_SUCCESS)
    {
        throw std::runtime_error("vkQueueSubmit failed.");
    }

    // 7) Present.
    const VkSwapchainKHR swapChains[] = {swapChain};
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores    = signalSemaphores;
    presentInfo.swapchainCount     = 1;
    presentInfo.pSwapchains        = swapChains;
    presentInfo.pImageIndices      = &imageIndex;

    const VkResult presentResult = vkQueuePresentKHR(presentQueue, &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR ||
        presentResult == VK_SUBOPTIMAL_KHR         ||
        framebufferResized)
    {
        framebufferResized = false;
        recreateSwapChain();
    }
    else if (presentResult != VK_SUCCESS)
    {
        throw std::runtime_error("vkQueuePresentKHR failed.");
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

    // 8) FPS counter — update title every second.
    frameCount++;
    const double now = glfwGetTime();
    if (now - lastFpsTime >= 1.0)
    {
        const double fps = static_cast<double>(frameCount) / (now - lastFpsTime);
        const std::string title =
            std::string(kWindowTitle) +
            "  |  FPS: " + std::to_string(static_cast<int>(fps));
        glfwSetWindowTitle(window, title.c_str());
        frameCount  = 0;
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
    validationFeatures.pEnabledValidationFeatures   = &bestPracticesEnable;
    validationFeatures.pNext                         = &debugCreateInfo;

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
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

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

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
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
        VkImageView attachments[] = {swapChainImageViews[i]};

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
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

    VkClearValue clearColor = {{{0.1f, 0.1f, 0.2f, 1.0f}}};
    // Red test color (manual toggle): VkClearValue clearColor = {{{1.0f, 0.0f, 0.0f, 1.0f}}};
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(cmdBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    VkViewport viewport{};
    viewport.x        = 0.0f;
    viewport.y        = 0.0f;
    viewport.width    = static_cast<float>(swapChainExtent.width);
    viewport.height   = static_cast<float>(swapChainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapChainExtent;
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    vkCmdDraw(cmdBuffer, 3, 1, 0, 0);

    vkCmdEndRenderPass(cmdBuffer);

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
    createInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode    = reinterpret_cast<const uint32_t *>(code.data());

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
    const std::vector<char> vertCode = readSpvFile(shaderDir + "triangle.vert.spv");
    const std::vector<char> fragCode = readSpvFile(shaderDir + "triangle.frag.spv");

    VkShaderModule vertShaderModule = createShaderModule(vertCode);
    VkShaderModule fragShaderModule = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo vertStageInfo{};
    vertStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStageInfo.stage  = VK_SHADER_STAGE_VERTEX_BIT;
    vertStageInfo.module = vertShaderModule;
    vertStageInfo.pName  = "main";

    VkPipelineShaderStageCreateInfo fragStageInfo{};
    fragStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStageInfo.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStageInfo.module = fragShaderModule;
    fragStageInfo.pName  = "main";

    const VkPipelineShaderStageCreateInfo shaderStages[] = {
        vertStageInfo, fragStageInfo};

    // No vertex buffers — positions and colors are hardcoded in the vertex shader.
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount   = 0;
    vertexInputInfo.pVertexBindingDescriptions      = nullptr;
    vertexInputInfo.vertexAttributeDescriptionCount = 0;
    vertexInputInfo.pVertexAttributeDescriptions    = nullptr;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // Dynamic viewport and scissor — set per-frame in recordCommandBuffer().
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports    = nullptr;  // set dynamically
    viewportState.scissorCount  = 1;
    viewportState.pScissors     = nullptr;  // set dynamically

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable        = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode             = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth               = 1.0f;
    rasterizer.cullMode                = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace               = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable         = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable  = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable   = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments    = &colorBlendAttachment;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount          = 0;
    pipelineLayoutInfo.pushConstantRangeCount  = 0;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                               &pipelineLayout) != VK_SUCCESS)
    {
        // Clean up shader modules before propagating the error.
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
        throw std::runtime_error("Failed to create pipeline layout.");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount          = 2;
    pipelineInfo.pStages             = shaderStages;
    pipelineInfo.pVertexInputState   = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState      = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState   = &multisampling;
    pipelineInfo.pDepthStencilState  = nullptr;
    pipelineInfo.pColorBlendState    = &colorBlending;
    const VkDynamicState dynamicStates[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicStateInfo{};
    dynamicStateInfo.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicStateInfo.dynamicStateCount = 2;
    dynamicStateInfo.pDynamicStates    = dynamicStates;

    pipelineInfo.pDynamicState = &dynamicStateInfo;
    pipelineInfo.layout              = pipelineLayout;
    pipelineInfo.renderPass          = renderPass;
    pipelineInfo.subpass             = 0;
    pipelineInfo.basePipelineHandle  = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex   = -1;

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

int main()
{
    try
    {
        SplatCoreApp app;
        app.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Fatal] " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
