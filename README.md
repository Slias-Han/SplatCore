# SplatCore

SplatCore is an industrial-oriented Vulkan renderer for 3D Gaussian Splatting, implemented in modern C++.

The current repository is still in the infrastructure stage: the active render path is a deterministic point-sprite pipeline, while the testing stack already covers VRAM lifecycle validation, GPU-side poison tests, and GPU-side SHA-256 consistency measurement for future non-determinism analysis.

## Current Status

- `v0.5`: VMA-based 3-region memory partitioning and VRAM logging
- `v0.6`: GPU VRAM poison test infrastructure, offscreen render target, CPU readback path, reusable `PoisonTestHarness`
- `v0.7`: GPU-side SHA-256 hash probe, 100-frame consistency analysis, root-cause report generation, CI integration

Current `v0.7` result on the shipped point-sprite path:
- `sha256_consistency_test`: `100.0%` consistency
- Interpretation: the current non-alpha point rendering path is deterministic
- Follow-up: rerun the same test after full 3DGS alpha blending is introduced

## Repository Layout

- [main.cpp](/d:/GitHubProjects/SplatCore_SilasHan/main.cpp): engine bootstrap, Vulkan setup, render loop, testing hooks
- [shaders/](/d:/GitHubProjects/SplatCore_SilasHan/shaders): graphics and compute shaders
- [src/core/memory/](/d:/GitHubProjects/SplatCore_SilasHan/src/core/memory): memory system and VRAM logger
- [src/tests/](/d:/GitHubProjects/SplatCore_SilasHan/src/tests): reusable GPU test harnesses (`PoisonTestHarness`, `HashProbe`)
- [tests/](/d:/GitHubProjects/SplatCore_SilasHan/tests): standalone regression and acceptance tests
- [docs/](/d:/GitHubProjects/SplatCore_SilasHan/docs): design notes and test constitutions

## Build

Windows, CMake, Vulkan SDK, GLFW, GLM, and a working MSVC toolchain are expected.

Configure:

```powershell
cmake -B build -S .
```

Build the main executable:

```powershell
cmake --build build --config Release --target SplatCore_SilasHan
```

Build shaders only:

```powershell
cmake --build build --config Release --target SplatCore_Shaders
```

## Run

Run the engine with a `.ply` scene:

> **Note**: Large scene files (>50 MB) are not tracked in git.
> Place them under `tests/assets/` locally.

```powershell
.\build\SplatCore_SilasHan.exe .\tests\assets\poison_test_scene.ply
```

Useful environment variables:

- `SPLATCORE_MAX_FRAMES`: stop automatically after `N` frames
- `SPLATCORE_HASH_PROBE=1`: enable GPU-side SHA-256 capture for 100 frames

Example:

```powershell
$env:SPLATCORE_HASH_PROBE='1'
$env:SPLATCORE_MAX_FRAMES='100'
.\build\SplatCore_SilasHan.exe .\tests\assets\poison_test_scene.ply
Remove-Item Env:SPLATCORE_HASH_PROBE, Env:SPLATCORE_MAX_FRAMES
```

When hash probe mode is enabled, the engine generates:

- `sha256_log.txt`
- `sha256_rootcause.md`

## Tests

Available executables:

- `SplatCore_MemoryLifecycleTests`: v0.5 memory lifecycle regression
- `SplatCore_PoisonTests`: v0.6 VRAM poison acceptance test
- `SplatCore_SHA256Tests`: v0.7 SHA-256 consistency acceptance test

Build all test targets:

```powershell
cmake --build build --config Release --target SplatCore_MemoryLifecycleTests
cmake --build build --config Release --target SplatCore_PoisonTests
cmake --build build --config Release --target SplatCore_SHA256Tests
```

Run them:

```powershell
.\build\SplatCore_MemoryLifecycleTests.exe
.\build\SplatCore_PoisonTests.exe .\tests\assets\poison_test_scene.ply .\build\shaders
.\build\SplatCore_SHA256Tests.exe .\tests\assets\poison_test_scene.ply .\build\shaders
```

Acceptance rule for `SplatCore_SHA256Tests`:

- return `0`: SHA-256 infrastructure is healthy (`consistencyRate > 0`)
- return `1`: infrastructure failure (`consistencyRate == 0`)

`consistencyRate < 100%` is allowed and is treated as a measurement outcome, not as a test failure.

## v0.7 Determinism Measurement

`v0.7` is about measuring non-determinism, not hiding it.

What it does:

- computes SHA-256 on the GPU with [shaders/sha256_compute.comp](/d:/GitHubProjects/SplatCore_SilasHan/shaders/sha256_compute.comp)
- captures one hash per frame for 100 frames
- writes a frame-by-frame log to `sha256_log.txt`
- emits an automatic summary report to `sha256_rootcause.md`

Key implementation pieces:

- [src/tests/HashProbe.h](/d:/GitHubProjects/SplatCore_SilasHan/src/tests/HashProbe.h)
- [src/tests/HashProbe.cpp](/d:/GitHubProjects/SplatCore_SilasHan/src/tests/HashProbe.cpp)
- [tests/sha256/test_sha256_consistency.cpp](/d:/GitHubProjects/SplatCore_SilasHan/tests/sha256/test_sha256_consistency.cpp)
- [docs/V07_SHA256_DETERMINISM_ANALYSIS.md](/d:/GitHubProjects/SplatCore_SilasHan/docs/V07_SHA256_DETERMINISM_ANALYSIS.md)

Current conclusion:

- the present point-sprite pipeline does not exhibit observable frame-to-frame hash divergence
- this does not prove future 3DGS alpha blending will be deterministic
- the same harness should be rerun once true over-compositing and parallel blending are introduced

## CI

The GitHub Actions workflow is:

- [.github/workflows/vram_poison_ci.yml](/d:/GitHubProjects/SplatCore_SilasHan/.github/workflows/vram_poison_ci.yml)
- [.github/workflows/build_check.yml](/d:/GitHubProjects/SplatCore_SilasHan/.github/workflows/build_check.yml)

It currently builds and runs:

- shader compilation
- VRAM poison regression
- SHA-256 consistency regression

It also uploads SHA-256 artifacts on every run:

- `build/sha256_log.txt`
- `build/sha256_rootcause.md`

Recent Linux CI hardening on `ubuntu-22.04`:

- replaced the broken `apt install glslc` path with LunarG Vulkan SDK packages so shader compilation works reliably in GitHub-hosted runners
- added Linux platform guards in CMake and kept Win32-only logic behind `if(WIN32)` so `Configure CMake` and Linux builds no longer trip on MSVC-specific settings
- switched Linux Vulkan surface defines to `VK_USE_PLATFORM_XLIB_KHR` and added `libx11-dev`
- wrapped the runtime Vulkan tests with `xvfb-run -a`, because `llvmpipe` alone is not enough for GLFW/swapchain-based tests in headless CI

Current CI baseline:

- `Build Check` passes on Ubuntu 22.04
- `VRAM Poison Test (M1 Gate)` passes on Ubuntu 22.04 with `llvmpipe + xvfb`
- `SplatCore_PoisonTests`: `4/4 PASS`
- `SplatCore_SHA256Tests`: passes and uploads `sha256_log.txt` / `sha256_rootcause.md`

## Changelog

### v0.7 - SHA-256 VRAM bitwise consistency measurement

- Added GPU-side SHA-256 compute shader and `HashProbe`
- Added `SplatCore_SHA256Tests` acceptance test
- Added automatic `sha256_log.txt` and `sha256_rootcause.md` generation
- Added main-engine hash probe mode via `SPLATCORE_HASH_PROBE=1`
- Added CI steps for SHA-256 build, run, and artifact upload

### v0.6 - VRAM poison test infrastructure

- Added offscreen render target and CPU readback path
- Added reusable `PoisonTestHarness`
- Added VRAM poison regression test with `4/4 PASS`
- Design note: [docs/V06_VRAM_POISON_TEST_DESIGN.md](/d:/GitHubProjects/SplatCore_SilasHan/docs/V06_VRAM_POISON_TEST_DESIGN.md)

### v0.5 - Memory lifecycle hardening

- Added VMA memory lifecycle constitution
- Added rollback-safe Vulkan buffer/image creation paths
- Added explicit uniform buffer unmap and unified teardown helpers
- Design note: [docs/V05_MEMORY_LIFECYCLE_CONSTITUTION.md](/d:/GitHubProjects/SplatCore_SilasHan/docs/V05_MEMORY_LIFECYCLE_CONSTITUTION.md)

### v0.4 - Cube rendering

- Vertex buffer, index buffer, MVP matrix, rotating 3D cube

### v0.3 - Resize stabilization

- Resize stabilization, BestPractices clean run, FPS display

### v0.2 - Gradient triangle

- RGB gradient triangle and graphics pipeline

### v0.1 - Hello Vulkan

- Object-oriented Vulkan bootstrap with GLFW
- Instance, validation layer, debug messenger, surface, device, swapchain, render pass, framebuffers, command pool/buffer, sync primitives
- Clear-only render pass and clean reverse-order shutdown
