#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <functional>
#include <vector>

struct EpsilonRunResult {
    int poseIndex = -1;
    int directionIndex = -1;    // 0-7, see direction definitions below
    float epsilon = 0.0f;
    float maxDepthJump = -1.0f;
    float frobeniusNorm = -1.0f;
    float discontinuityRate = 1.0f;
    bool pass = false;
};

struct EpsilonReport {
    float epsilon = 0.0f;
    int posesCount = 0;
    int directionsCount = 0;
    int totalRuns = 0;
    std::vector<EpsilonRunResult> runs;
    EpsilonRunResult worstCase{};
    bool overallPass = false;
};

// Perturbation direction indices (used in EpsilonRunResult.directionIndex):
//   0: +x translation   1: -x translation
//   2: +y translation   3: -y translation
//   4: +z translation   5: -z translation
//   6: +pitch rotation  7: -pitch rotation

class EpsilonProbe {
public:
    static constexpr float JUMP_THRESHOLD = 1.0f;
    static constexpr float DISC_RATE_LIMIT = 0.0001f;

    using RenderDepthFn = std::function<std::vector<float>(const glm::mat4& view)>;

    explicit EpsilonProbe(RenderDepthFn renderDepth,
                          uint32_t imageWidth,
                          uint32_t imageHeight);

    EpsilonRunResult runOne(int poseIndex,
                            const glm::mat4& refView,
                            int directionIndex,
                            float epsilon = 1.192e-7f);

    std::vector<EpsilonRunResult> runPose(int poseIndex,
                                          const glm::mat4& refView,
                                          float epsilon = 1.192e-7f);

private:
    RenderDepthFn m_renderDepth;
    uint32_t m_width = 0;
    uint32_t m_height = 0;

    glm::mat4 applyPerturbation(const glm::mat4& view,
                                int directionIndex,
                                float epsilon) const;

    std::vector<float> computeDeltaD(const std::vector<float>& ref,
                                     const std::vector<float>& perturbed) const;
};
