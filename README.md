# SplatCore
A high-performance, hardware-accelerated 3D Gaussian Splatting engine engineered in modern C++.

## Changelog

### v0.1 - Hello Vulkan
- Implemented object-oriented Vulkan bootstrap (`SplatCoreApp`) with GLFW window creation.
- Added Vulkan initialization chain: Instance, Validation Layer (`VK_LAYER_KHRONOS_validation`), Debug Utils Messenger, Surface, Physical/Logical Device, Swapchain, Image Views, Render Pass, Framebuffers, Command Pool/Buffer, and sync primitives.
- Implemented minimal render loop with strict frame synchronization (Semaphore + Fence) and swapchain present flow.
- Added clear-only render pass (no geometry) using deep-blue clear color:
  - `VkClearValue clearColor = {{{ 0.1f, 0.1f, 0.2f, 1.0f }}};`
- Ensured clean shutdown with `vkDeviceWaitIdle` and reverse-order destruction of all Vulkan resources.
- Validation Layer policy for v0.1: zero warnings and zero errors during normal run.
