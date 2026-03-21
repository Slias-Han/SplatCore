#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "../../src/tests/HashProbe.h"

#define SPLATCORE_NO_ENTRYPOINT
// TODO(P1): same as test_vram_poison.cpp - depends on engine core extraction
#include "../../main.cpp"

namespace {

std::string buildSha256ShaderPath(const char* spvDir)
{
    const char* dir = (spvDir != nullptr && spvDir[0] != '\0') ? spvDir : "shaders";
    std::string path = dir;
    if (!path.empty() && path.back() != '/' && path.back() != '\\')
    {
        path.push_back('/');
    }
    path += "sha256_compute.spv";
    return path;
}

bool fileExists(const char* path)
{
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

bool fileContains(const char* path, const char* needle)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.good())
    {
        return false;
    }

    std::string contents((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    return contents.find(needle) != std::string::npos;
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::fprintf(stderr,
                     "Usage: SplatCore_SHA256Tests <path_to_file.ply> [spv_dir]\n");
        return 1;
    }

    const char* plyPath = argv[1];
    const char* spvDir = (argc >= 3) ? argv[2] : "shaders";
    const std::string shaderPath = buildSha256ShaderPath(spvDir);

    SplatCoreApp app;
    try
    {
        app.setPlyPath(plyPath);
        app.initializeWindowForTesting();
        app.initializeVulkanCoreForTesting();
        app.enableHashProbeForTesting(shaderPath);
        app.initializeRenderResourcesForTesting();
        app.renderFramesForTesting(100);

        const std::vector<SplatCore::FrameHash>& hashes = app.frameHashesForTesting();
        SplatCore::HashProbe reporter;
        const float consistencyRate =
            reporter.analyzeConsistency(hashes, "sha256_log.txt");
        reporter.generateRootCauseReport(hashes, "sha256_rootcause.md");

        std::printf("\n=== SHA-256 一致性测试结果 ===\n");
        std::printf("100 帧哈希一致率: %.1f%%\n", consistencyRate * 100.0f);
        std::printf("详细日志: sha256_log.txt\n");
        std::printf("根因分析: sha256_rootcause.md\n");

        if (!fileExists("sha256_rootcause.md"))
        {
            std::printf("[FAIL] 未生成 sha256_rootcause.md\n");
            app.shutdownForTesting();
            return 1;
        }
        if (!fileContains("sha256_rootcause.md",
                          "Known Non-Determinism Sources"))
        {
            std::printf("[FAIL] 根因报告缺少 Known Non-Determinism Sources section\n");
            app.shutdownForTesting();
            return 1;
        }
        if (!fileContains("sha256_rootcause.md",
                          "Expected Non-Determinism Sources"))
        {
            std::printf("[FAIL] 根因报告缺少 Expected Non-Determinism Sources section\n");
            app.shutdownForTesting();
            return 1;
        }
        if (!fileContains("sha256_rootcause.md", "depthConsistency:"))
        {
            std::printf("[FAIL] 根因报告缺少 depthConsistency 行\n");
            app.shutdownForTesting();
            return 1;
        }

        app.shutdownForTesting();

        if (consistencyRate == 0.0f)
        {
            std::printf("[FAIL] 一致率为 0，SHA-256 基础设施异常\n");
            return 1;
        }

        std::printf("[PASS] SHA-256 基础设施运行正常，一致率: %.1f%%\n",
                    consistencyRate * 100.0f);
        std::printf("       （一致率 < 100%% 是预期的，根因见 sha256_rootcause.md）\n");
        return 0;
    }
    catch (const std::exception& e)
    {
        app.shutdownForTesting();
        std::fprintf(stderr, "[FAIL] SHA-256 一致性测试异常: %s\n", e.what());
        return 1;
    }
}
