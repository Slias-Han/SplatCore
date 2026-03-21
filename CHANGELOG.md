# Changelog

This file tracks user-visible engineering milestones and validation-relevant changes.

## Unreleased

### Test hygiene and v0.6 compliance hardening

- Added tracked-allocation poisoning for STATIC + DYNAMIC regions through `MemorySystem::getAllocations()`
- Reworked `SplatCore_PoisonTests` so baseline and poisoned replay reuse a single `VkDevice`
- Added poison log output in the form `Poisoning N allocations (STATIC: X, DYNAMIC: Y)`
- Added a real child-process death test for STATIC over-budget allocation in `SplatCore_MemoryLifecycleTests`
- Marked poison-test vertex buffers with `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` so Debug validation remains clean
- Kept existing regression targets green: memory lifecycle, poison test, and SHA-256 consistency test

## v0.7 - SHA-256 VRAM bitwise consistency measurement

- Added GPU-side SHA-256 compute shader and `HashProbe`
- Added `SplatCore_SHA256Tests` acceptance test
- Added automatic `sha256_log.txt` and `sha256_rootcause.md` generation
- Added main-engine hash probe mode via `SPLATCORE_HASH_PROBE=1`
- Added CI steps for SHA-256 build, run, and artifact upload

## v0.6 - VRAM poison test infrastructure

- Added offscreen render target and CPU readback path
- Added reusable `PoisonTestHarness`
- Added VRAM poison regression test with `4/4 PASS`
- Added Linux CI coverage for poison and SHA-256 paths

## v0.5 - Memory lifecycle hardening

- Added VMA memory lifecycle constitution
- Added rollback-safe Vulkan buffer/image creation paths
- Added explicit uniform buffer unmap and unified teardown helpers
- Added memory lifecycle regression coverage for dynamic, staging, and death-test behavior

## v0.4 - Cube rendering

- Vertex buffer, index buffer, MVP matrix, rotating 3D cube

## v0.3 - Resize stabilization

- Resize stabilization, BestPractices clean run, FPS display

## v0.2 - Gradient triangle

- RGB gradient triangle and graphics pipeline

## v0.1 - Hello Vulkan

- Object-oriented Vulkan bootstrap with GLFW
- Instance, validation layer, debug messenger, surface, device, swapchain, render pass, framebuffers, command pool/buffer, sync primitives
- Clear-only render pass and clean reverse-order shutdown
