#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../../src/tests/EpsilonProbe.h"

#define SPLATCORE_NO_ENTRYPOINT
#include "../../main.cpp"

namespace {

constexpr float kDefaultEpsilon = 1.192e-7f;

const char* directionLabel(int directionIndex)
{
    static constexpr const char* kLabels[8] = {
        "+x trans",
        "-x trans",
        "+y trans",
        "-y trans",
        "+z trans",
        "-z trans",
        "+pitch rot",
        "-pitch rot"};

    return (directionIndex >= 0 && directionIndex < 8) ? kLabels[directionIndex]
                                                        : "unknown";
}

const char* gitShaOrLocal()
{
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4996)
#endif
    const char* gitSha = std::getenv("GITHUB_SHA");
    const char* posesEnv = std::getenv("EPSILON_TEST_POSES");
#ifdef _MSC_VER
#pragma warning(pop)
#endif

    (void)posesEnv;
    return (gitSha != nullptr && gitSha[0] != '\0') ? gitSha : "local";
}

int epsilonTestPoses()
{
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4996)
#endif
    const char* posesEnv = std::getenv("EPSILON_TEST_POSES");
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    if (posesEnv == nullptr || posesEnv[0] == '\0')
    {
        return 10;
    }

    try
    {
        const int parsed = std::stoi(posesEnv);
        return parsed > 0 ? parsed : 10;
    }
    catch (...)
    {
        return 10;
    }
}

std::string generatedTimestamp()
{
    std::time_t now = std::time(nullptr);
    std::tm localTime{};
#ifdef _WIN32
    localtime_s(&localTime, &now);
#else
    localtime_r(&now, &localTime);
#endif

    std::ostringstream oss;
    oss << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

bool isFiniteVec3(const glm::vec3& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

std::vector<glm::mat4> buildReferenceViews(const glm::vec3& center,
                                           float radius,
                                           int posesCount)
{
    std::vector<glm::mat4> views;
    views.reserve(static_cast<size_t>(posesCount));

    static constexpr float kGoldenAngle = 2.39996322972865332f;
    const std::array<glm::vec3, 3> upAxes = {
        glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, 1.0f)};

    for (int i = 0; i < posesCount; ++i)
    {
        const float t = (static_cast<float>(i) + 0.5f) / static_cast<float>(posesCount);
        const float y = 1.0f - 2.0f * t;
        const float radial = std::sqrt(std::max(0.0f, 1.0f - y * y));
        const float theta = kGoldenAngle * static_cast<float>(i);

        glm::vec3 offset(
            std::cos(theta) * radial,
            y,
            std::sin(theta) * radial);
        offset = glm::normalize(offset);

        const glm::vec3 eye = center + radius * offset;
        const glm::vec3 forward = glm::normalize(center - eye);

        glm::vec3 up = upAxes[static_cast<size_t>(i % static_cast<int>(upAxes.size()))];
        for (size_t attempt = 0; attempt < upAxes.size(); ++attempt)
        {
            const glm::vec3 candidate =
                upAxes[(static_cast<size_t>(i) + attempt) % upAxes.size()];
            if (std::abs(glm::dot(forward, candidate)) < 0.95f)
            {
                up = candidate;
                break;
            }
        }

        views.push_back(glm::lookAt(eye, center, up));
    }

    return views;
}

bool writeEpsilonReport(const EpsilonReport& report)
{
    std::ofstream file("epsilon_report.md", std::ios::binary);
    if (!file.good())
    {
        return false;
    }

    file << "# v0.8 Epsilon Perturbation Invariance Report\n";
    file << "Generated: " << generatedTimestamp() << "\n";
    file << "Commit: " << gitShaOrLocal() << "\n\n";

    file << "## Configuration\n";
    file << "| Parameter         | Value                     |\n";
    file << "|-------------------|---------------------------|\n";
    file << "| Epsilon           | 1.192e-07 (FP32 ε)        |\n";
    file << "| Jump threshold    | 1.0 m                     |\n";
    file << "| Pass limit        | < 0.01% discontinuity     |\n";
    file << "| Poses tested      | " << report.posesCount << "                        |\n";
    file << "| Directions / pose | 8 (+x -x +y -y +z -z +pitch -pitch) |\n";
    file << "| Total runs        | " << report.totalRuns << "                        |\n";
    file << "| Render path       | point-sprite (no depth sort) |\n\n";

    file << "## Per-run results\n";
    file << "| Pose | Direction   | maxJump (m) | discRate   | Pass |\n";
    file << "|------|-------------|-------------|------------|------|\n";
    file << std::fixed << std::setprecision(6);
    for (const EpsilonRunResult& run : report.runs)
    {
        file << "| "
             << std::setw(2) << run.poseIndex << "   "
             << "| " << std::left << std::setw(11) << directionLabel(run.directionIndex)
             << std::right << "| "
             << std::setw(11) << run.maxDepthJump << " | "
             << std::setw(8) << std::setprecision(4)
             << (run.discontinuityRate * 100.0f) << "% | "
             << (run.pass ? "YES " : "NO  ") << " |\n"
             << std::setprecision(6);
    }
    file << "\n";

    file << "## Worst case\n";
    file << "Pose " << report.worstCase.poseIndex
         << ", direction " << directionLabel(report.worstCase.directionIndex)
         << ": discontinuityRate = " << std::setprecision(4)
         << (report.worstCase.discontinuityRate * 100.0f) << "%\n";
    file << std::setprecision(4)
         << "maxDepthJump = " << report.worstCase.maxDepthJump << " m\n\n";

    const int passCount = static_cast<int>(std::count_if(
        report.runs.begin(),
        report.runs.end(),
        [](const EpsilonRunResult& run) { return run.pass; }));
    const int failCount = report.totalRuns - passCount;

    file << "## Summary\n";
    file << "PASS: [" << passCount << "]/" << report.totalRuns << " runs passed\n";
    file << "FAIL: [" << failCount << "]/" << report.totalRuns
         << " runs exceeded 0.01% threshold\n\n";

    file << "## Conclusion\n";
    file << (report.overallPass ? "PASS" : "FAIL") << "\n\n";

    file << "## Interpretation\n";
    file << "Current render path: point-sprite (deterministic, no parallel sort)\n";
    file << "Expected result on this path: all runs PASS (no depth sort = no sort-flip\n";
    file << "  discontinuities possible)\n";
    file << "⚠ This result does NOT confirm M1 milestone.\n";
    file << "Rerun required after:\n";
    file << "  - v1.2: GPU depth sort (radix sort) introduced - sort-flip risk begins\n";
    file << "  - v2.0: alpha blending over-compositing - atomicAdd non-determinism risk begins\n";

    return file.good();
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::fprintf(stderr,
                     "Usage: SplatCore_EpsilonTests <path_to_file.ply> [spv_dir]\n");
        return 1;
    }

    const char* plyPath = argv[1];
    [[maybe_unused]] const char* spvDir = (argc >= 3) ? argv[2] : "shaders";

    SplatCoreApp app;
    try
    {
        app.setPlyPath(plyPath);
        app.initializeWindowForTesting();
        app.initializeVulkanCoreForTesting();
        app.initializeRenderResourcesForTesting();

        if (!app.sceneBoundsValidForTesting() ||
            !isFiniteVec3(app.sceneBoundsMinForTesting()) ||
            !isFiniteVec3(app.sceneBoundsMaxForTesting()))
        {
            std::fprintf(stderr, "[FAIL] Invalid scene bounds for epsilon test.\n");
            app.shutdownForTesting();
            return 1;
        }

        const glm::vec3 boundsMin = app.sceneBoundsMinForTesting();
        const glm::vec3 boundsMax = app.sceneBoundsMaxForTesting();
        const glm::vec3 center = 0.5f * (boundsMin + boundsMax);
        const glm::vec3 extent = boundsMax - boundsMin;
        const float radius = std::max(0.5f * glm::length(extent), 1.0f);
        const int posesCount = epsilonTestPoses();
        const std::vector<glm::mat4> referenceViews =
            buildReferenceViews(center, radius, posesCount);

        const VkExtent2D extent2D = app.renderExtentForTesting();
        const EpsilonProbe::RenderDepthFn renderDepth =
            [&app](const glm::mat4& view) -> std::vector<float>
        {
            std::vector<float> depth;
            try
            {
                app.renderDepthForTesting(view, depth);
            }
            catch (const std::exception& e)
            {
                std::fprintf(stderr,
                             "[FAIL] renderDepthFn exception: %s\n",
                             e.what());
                depth.clear();
            }
            return depth;
        };

        EpsilonProbe probe(renderDepth, extent2D.width, extent2D.height);
        EpsilonReport report{};
        report.epsilon = kDefaultEpsilon;
        report.posesCount = posesCount;
        report.directionsCount = 8;
        report.totalRuns = posesCount * report.directionsCount;

        for (int poseIndex = 0; poseIndex < posesCount; ++poseIndex)
        {
            const std::vector<EpsilonRunResult> poseRuns =
                probe.runPose(poseIndex, referenceViews[static_cast<size_t>(poseIndex)],
                              kDefaultEpsilon);
            report.runs.insert(report.runs.end(), poseRuns.begin(), poseRuns.end());
        }

        report.overallPass = std::all_of(
            report.runs.begin(),
            report.runs.end(),
            [](const EpsilonRunResult& run) { return run.pass; });

        if (!report.runs.empty())
        {
            report.worstCase = *std::max_element(
                report.runs.begin(),
                report.runs.end(),
                [](const EpsilonRunResult& a, const EpsilonRunResult& b)
                {
                    if (a.discontinuityRate == b.discontinuityRate)
                    {
                        return a.maxDepthJump < b.maxDepthJump;
                    }
                    return a.discontinuityRate < b.discontinuityRate;
                });
        }

        if (!writeEpsilonReport(report))
        {
            std::fprintf(stderr, "[FAIL] Failed to write epsilon_report.md\n");
            app.shutdownForTesting();
            return 1;
        }

        const int passCount = static_cast<int>(std::count_if(
            report.runs.begin(),
            report.runs.end(),
            [](const EpsilonRunResult& run) { return run.pass; }));

        std::printf("\n=== Epsilon 扰动拓扑不变性测试结果 ===\n");
        std::printf("Poses: %d, runs: %d, pass: %d\n",
                    report.posesCount,
                    report.totalRuns,
                    passCount);
        std::printf("报告: epsilon_report.md\n");

        app.shutdownForTesting();
        return report.overallPass ? 0 : 1;
    }
    catch (const std::exception& e)
    {
        app.shutdownForTesting();
        std::fprintf(stderr, "[FAIL] EpsilonTests exception: %s\n", e.what());
        return 1;
    }
}
