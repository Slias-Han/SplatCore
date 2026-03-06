# SplatCore v0.5 Memory Lifecycle Constitution

## 1. Scope
This constitution governs all CPU/GPU memory and Vulkan resource ownership in `SplatCoreApp` (v0.5).

## 2. Core Laws
1. Single owner: every allocation has exactly one owner field or container.
2. Explicit destroy path: every owner must have one deterministic release function.
3. Reverse-order teardown: parent scope must outlive child scope.
4. Failure rollback: partial creation must self-clean before throwing.
5. Reload safety: re-create paths must release old ownership first.

## 3. Ownership Ledger
| Resource | Owner | Create Path | Destroy Path | Scope |
|---|---|---|---|---|
| `VkInstance` | `instance` | `createInstance()` | `cleanup()` | process |
| `VkSurfaceKHR` | `surface` | `createSurface()` | `cleanup()` | process |
| `VkDevice` | `device` | `createLogicalDevice()` | `cleanup()` | process |
| Swapchain + image views | `swapChain`, `swapChainImageViews` | `createSwapChain()`, `createImageViews()` | `cleanupSwapChain()` | resize epoch |
| Depth image + memory + view | `depthImage`, `depthImageMemory`, `depthImageView` | `createDepthResources()` | `cleanupSwapChain()` | resize epoch |
| Command pool/buffers | `commandPool`, `commandBuffers` | `createCommandPool()`, `createCommandBuffer()` | `cleanup()` | device |
| Point vertex buffer + memory | `pointVertexBuffer`, `pointVertexBufferMemory` | `loadPointCloud()` | `cleanup()` / reload pre-release | scene |
| Uniform buffers + memory + map ptr | `uniformBuffers*` vectors | `createUniformBuffers()` | `cleanup()` (explicit unmap then free) | frame slot |

## 4. Failure Rollback Contract
1. `createBuffer()`:
   - if raw Vulkan memory allocation fails: destroy created buffer before throw.
   - if `vkBindBufferMemory` fails: free memory + destroy buffer before throw.
2. `createImage()`:
   - if allocation fails: destroy image before throw.
   - if bind fails: free memory + destroy image before throw.
3. `createDepthResources()`:
   - if image view creation fails: free depth image + memory before throw.
4. `loadPointCloud()`:
   - staging and destination GPU buffers are both rolled back on any exception.

## 5. Whiteboard Merge Gate
A PR that introduces any new allocation is blocked unless it includes:
1. Owner field/container.
2. Create function.
3. Destroy function (or consolidated helper).
4. Failure rollback branch.
5. Resize/reload behavior (if applicable).

## 6. Canonical Helper APIs (v0.5)
- `destroyBufferAndMemory(VkBuffer&, VkDeviceMemory&)`
- `destroyImageAndMemory(VkImage&, VkDeviceMemory&)`

These helpers are mandatory for paired resource teardown to avoid split ownership.

## 7. VMA Binding Path Note
Buffer/Image use the `vkCreate* + vmaAllocateMemory + vmaBind*Memory` path instead of
`vmaCreateBuffer` / `vmaCreateImage`.

Reason: the convenience APIs can hang under the current Debug Validation Layer combination.
This is treated as a known driver-compatibility issue, not an ownership-model exception.

Memory ownership remains 100% under VMA, so the constitution core constraints are unchanged.
