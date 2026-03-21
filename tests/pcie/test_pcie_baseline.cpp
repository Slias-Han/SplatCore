#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../../src/tests/PcieTransferTracker.h"

#define SPLATCORE_NO_ENTRYPOINT
// TODO(P1): decouple from main.cpp — same as poison/sha256 tests
// Depends on SplatCoreApp engine core extraction to static lib
#include "../../main.cpp"

namespace {

int pcieTestFrames()
{
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4996)
#endif
    const char* framesEnv = std::getenv("PCIE_TEST_FRAMES");
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    if (framesEnv == nullptr || framesEnv[0] == '\0')
    {
        return 100;
    }

    try
    {
        const int parsed = std::stoi(framesEnv);
        return parsed > 0 ? parsed : 100;
    }
    catch (...)
    {
        return 100;
    }
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::fprintf(stderr,
                     "Usage: SplatCore_PcieTests <path_to_file.ply> [spv_dir]\n");
        return 1;
    }

    const char* plyPath = argv[1];
    [[maybe_unused]] const char* spvDir = (argc >= 3) ? argv[2] : "shaders";
    const int frames = pcieTestFrames();

    PcieTransferTracker::instance().reset();
    PcieTransferTracker::instance().setFramesMeasured(static_cast<uint32_t>(frames));

    SplatCoreApp app;
    try
    {
        app.setPlyPath(plyPath);
        app.initializeWindowForTesting();
        app.initializeVulkanCoreForTesting();
        app.initializeRenderResourcesForTesting();
        app.renderFramesForTesting(static_cast<uint32_t>(frames));

        PcieTransferTracker::instance().writeReport("pcie_baseline_report.md");

        const std::vector<TransferEvent> eliminatable =
            PcieTransferTracker::instance().eliminatableCopies();

        std::printf("\n=== PCIe 基线测试结果 ===\n");
        std::printf("Frames measured: %d\n", frames);
        std::printf("Eliminatable transfers: %zu\n", eliminatable.size());
        std::printf("Report: pcie_baseline_report.md\n");

        app.shutdownForTesting();

        if (eliminatable.size() < 3)
        {
            std::printf("[FAIL] 可消除拷贝少于 3 个，说明插桩不完整\n");
            return 1;
        }

        std::printf("[PASS] PCIe 基线探针运行正常，已识别 %zu 个可消除传输\n",
                    eliminatable.size());
        return 0;
    }
    catch (const std::exception& e)
    {
        app.shutdownForTesting();
        std::fprintf(stderr, "[FAIL] PCIe baseline test exception: %s\n", e.what());
        return 1;
    }
}
