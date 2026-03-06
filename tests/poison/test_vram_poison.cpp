#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../src/tests/PoisonTestHarness.h"

#define SPLATCORE_NO_ENTRYPOINT
#include "../../main.cpp"

namespace {

struct RunResult {
    std::vector<uint32_t> pixels;
    VkPhysicalDeviceProperties physicalDeviceProperties{};
};

std::string buildPoisonShaderPath(const char* spvDir)
{
    const char* dir = (spvDir != nullptr && spvDir[0] != '\0') ? spvDir : "shaders";
    std::string path = dir;
    if (!path.empty() && path.back() != '/' && path.back() != '\\')
    {
        path.push_back('/');
    }
    path += "vram_poison_comp.spv";
    return path;
}

bool samePhysicalDevice(const VkPhysicalDeviceProperties& a,
                        const VkPhysicalDeviceProperties& b)
{
    return a.vendorID == b.vendorID &&
           a.deviceID == b.deviceID &&
           a.deviceType == b.deviceType &&
           std::strcmp(a.deviceName, b.deviceName) == 0;
}

void savePixels(const char* path, const std::vector<uint32_t>& pixels)
{
    std::FILE* file = nullptr;
#ifdef _MSC_VER
    if (fopen_s(&file, path, "wb") != 0)
    {
        file = nullptr;
    }
#else
    file = std::fopen(path, "wb");
#endif
    if (file == nullptr)
    {
        std::fprintf(stderr, "[WARN] Failed to open %s for writing.\n", path);
        return;
    }

    const uint32_t pixelCount = static_cast<uint32_t>(pixels.size());
    const size_t countWritten = std::fwrite(&pixelCount, sizeof(uint32_t), 1, file);
    const size_t pixelsWritten =
        pixels.empty() ? 0 : std::fwrite(pixels.data(), sizeof(uint32_t), pixels.size(), file);
    if (countWritten != 1 || pixelsWritten != pixels.size())
    {
        std::fprintf(stderr, "[WARN] Failed to fully write %s.\n", path);
    }

    std::fclose(file);
}

int comparePixels(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b)
{
    if (a.size() != b.size())
    {
        return INT_MAX;
    }

    int diffCount = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (a[i] != b[i])
        {
            ++diffCount;
        }
    }

    return diffCount;
}

void saveDiffReport(const char* patternName,
                    const std::vector<uint32_t>& baseline,
                    const std::vector<uint32_t>& poisoned)
{
    const std::string path = std::string("diff_") + patternName + ".txt";

    std::FILE* file = nullptr;
#ifdef _MSC_VER
    if (fopen_s(&file, path.c_str(), "wb") != 0)
    {
        file = nullptr;
    }
#else
    file = std::fopen(path.c_str(), "wb");
#endif
    if (file == nullptr)
    {
        std::fprintf(stderr, "[WARN] Failed to open %s for writing.\n", path.c_str());
        return;
    }

    if (baseline.size() != poisoned.size())
    {
        std::fprintf(file,
                     "[size mismatch] baseline=%zu poisoned=%zu\n",
                     baseline.size(),
                     poisoned.size());
    }

    const size_t limit = std::min(baseline.size(), poisoned.size());
    size_t reported = 0;
    for (size_t i = 0; i < limit && reported < 32; ++i)
    {
        if (baseline[i] != poisoned[i])
        {
            std::fprintf(file,
                         "[pixel %6u] baseline=0x%08X  poisoned=0x%08X\n",
                         static_cast<unsigned>(i),
                         baseline[i],
                         poisoned[i]);
            ++reported;
        }
    }

    if (reported == 0 && baseline.size() != poisoned.size())
    {
        std::fprintf(file, "[no overlapping pixel diffs to report]\n");
    }

    std::fclose(file);
}

Allocation makeBufferAllocation(MemoryRegion region,
                                VkBufferUsageFlags usage,
                                VmaMemoryUsage vmaUsage,
                                VkDeviceSize size,
                                const char* name)
{
    AllocationDesc desc{};
    desc.region = region;
    desc.bufferUsage = usage;
    desc.vmaUsage = vmaUsage;
    desc.size = size;
    desc.allocationName = name;
    desc.imageInfo = nullptr;
    return MemorySystem::allocate(desc);
}

RunResult runBaseline(const char* plyPath)
{
    SplatCoreApp app;
    app.setPlyPath(plyPath);
    app.initializeWindowForTesting();
    app.initializeVulkanCoreForTesting();

    RunResult result{};
    result.physicalDeviceProperties = app.physicalDevicePropertiesForTesting();

    app.initializeRenderResourcesForTesting();
    app.renderFramesForTesting(100);
    app.readbackOffscreenForTesting(result.pixels);
    app.shutdownForTesting();

    return result;
}

RunResult runPoisoned(const char* plyPath,
                      const char* spvDir,
                      SplatCore::PoisonPattern pattern)
{
    SplatCoreApp app;
    SplatCore::PoisonTestHarness harness;
    Allocation stagingAllocation{};
    Allocation staticAllocation{};
    Allocation dynamicAllocation{};

    try
    {
        app.setPlyPath(plyPath);
        app.initializeWindowForTesting();
        app.initializeVulkanCoreForTesting();

        RunResult result{};
        result.physicalDeviceProperties = app.physicalDevicePropertiesForTesting();

        const std::string shaderPath = buildPoisonShaderPath(spvDir);
        harness.init(app.deviceForTesting(),
                     app.physicalDeviceForTesting(),
                     app.commandPoolForTesting(),
                     app.computeQueueForTesting(),
                     shaderPath.c_str());

        stagingAllocation = makeBufferAllocation(
            MemoryRegion::STAGING,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_ONLY,
            1u << 20,
            "PoisonTest::StagingScratch");
        harness.poisonBuffer(stagingAllocation.buffer, stagingAllocation.size, pattern);
        MemorySystem::free(stagingAllocation);

        staticAllocation = makeBufferAllocation(
            MemoryRegion::STATIC,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY,
            4u << 20,
            "PoisonTest::StaticScratch");
        harness.poisonBuffer(staticAllocation.buffer, staticAllocation.size, pattern);
        MemorySystem::free(staticAllocation);

        dynamicAllocation = makeBufferAllocation(
            MemoryRegion::DYNAMIC,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY,
            1u << 20,
            "PoisonTest::DynamicScratch");
        harness.poisonBuffer(dynamicAllocation.buffer, dynamicAllocation.size, pattern);

        app.initializeRenderResourcesForTesting();
        harness.poisonBuffer(app.offscreenReadbackBufferForTesting(),
                             app.offscreenReadbackBufferSizeForTesting(),
                             pattern);

        app.renderFramesForTesting(100);
        app.readbackOffscreenForTesting(result.pixels);

        harness.shutdown();
        app.shutdownForTesting();
        return result;
    }
    catch (...)
    {
        harness.shutdown();
        app.shutdownForTesting();
        throw;
    }
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::fprintf(stderr,
                     "Usage: SplatCore_PoisonTests <path_to_file.ply> [spv_dir]\n");
        return 1;
    }

    const char* plyPath = argv[1];
    const char* spvDir = (argc >= 3) ? argv[2] : "shaders";

    std::vector<uint32_t> baselinePixels;
    VkPhysicalDeviceProperties baselineProps{};

    try
    {
        const RunResult baseline = runBaseline(plyPath);
        baselinePixels = baseline.pixels;
        baselineProps = baseline.physicalDeviceProperties;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "[FAIL] Baseline run failed: %s\n", e.what());
        return 1;
    }

    savePixels("poison_test_baseline.bin", baselinePixels);

    const SplatCore::PoisonPattern patterns[] = {
        SplatCore::PoisonPattern::NAN_PATTERN,
        SplatCore::PoisonPattern::POS_INF,
        SplatCore::PoisonPattern::NEG_INF,
        SplatCore::PoisonPattern::DEAD_BEEF,
    };
    const char* patternNames[] = {"NaN", "+Inf", "-Inf", "0xDEADBEEF"};

    int failCount = 0;
    for (int i = 0; i < 4; ++i)
    {
        std::vector<uint32_t> poisonedPixels;

        try
        {
            const RunResult poisoned = runPoisoned(plyPath, spvDir, patterns[i]);
            poisonedPixels = poisoned.pixels;

            if (!samePhysicalDevice(baselineProps, poisoned.physicalDeviceProperties))
            {
                std::printf("[FAIL] Pattern %-12s — physical device mismatch\n",
                            patternNames[i]);
                saveDiffReport(patternNames[i], baselinePixels, poisonedPixels);
                ++failCount;
                continue;
            }
        }
        catch (const std::exception& e)
        {
            std::printf("[FAIL] Pattern %-12s — exception: %s\n",
                        patternNames[i],
                        e.what());
            saveDiffReport(patternNames[i], baselinePixels, poisonedPixels);
            ++failCount;
            continue;
        }

        const int diffCount = comparePixels(baselinePixels, poisonedPixels);
        if (diffCount == 0)
        {
            std::printf("[PASS] Pattern %-12s — 100 帧输出与 baseline 逐 uint32_t 一致\n",
                        patternNames[i]);
        }
        else
        {
            std::printf("[FAIL] Pattern %-12s — %d 像素差异\n",
                        patternNames[i],
                        diffCount);
            saveDiffReport(patternNames[i], baselinePixels, poisonedPixels);
            ++failCount;
        }
    }

    std::printf("\n=== VRAM 毒化测试结果: %d/4 通过 ===\n", 4 - failCount);
    return failCount == 0 ? 0 : 1;
}
