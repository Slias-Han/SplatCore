#include "EpsilonProbe.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace {

EpsilonRunResult makeFailureResult(int poseIndex,
                                   int directionIndex,
                                   float epsilon)
{
    EpsilonRunResult result{};
    result.poseIndex = poseIndex;
    result.directionIndex = directionIndex;
    result.epsilon = epsilon;
    result.maxDepthJump = -1.0f;
    result.frobeniusNorm = -1.0f;
    result.discontinuityRate = 1.0f;
    result.pass = false;
    return result;
}

} // namespace

EpsilonProbe::EpsilonProbe(RenderDepthFn renderDepth,
                           uint32_t imageWidth,
                           uint32_t imageHeight)
    : m_renderDepth(std::move(renderDepth)),
      m_width(imageWidth),
      m_height(imageHeight)
{
}

EpsilonRunResult EpsilonProbe::runOne(int poseIndex,
                                      const glm::mat4& refView,
                                      int directionIndex,
                                      float epsilon)
{
    if (!m_renderDepth || m_width == 0 || m_height == 0 ||
        directionIndex < 0 || directionIndex > 7)
    {
        return makeFailureResult(poseIndex, directionIndex, epsilon);
    }

    const std::vector<float> refDepth = m_renderDepth(refView);
    const std::vector<float> perturbedDepth =
        m_renderDepth(applyPerturbation(refView, directionIndex, epsilon));
    const size_t expectedCount =
        static_cast<size_t>(m_width) * static_cast<size_t>(m_height);

    if (refDepth.size() != expectedCount || perturbedDepth.size() != expectedCount)
    {
        return makeFailureResult(poseIndex, directionIndex, epsilon);
    }

    const std::vector<float> deltaD = computeDeltaD(refDepth, perturbedDepth);
    if (deltaD.size() != expectedCount)
    {
        return makeFailureResult(poseIndex, directionIndex, epsilon);
    }

    EpsilonRunResult result{};
    result.poseIndex = poseIndex;
    result.directionIndex = directionIndex;
    result.epsilon = epsilon;
    result.maxDepthJump = deltaD.empty()
                              ? 0.0f
                              : *std::max_element(deltaD.begin(), deltaD.end());

    double squaredSum = 0.0;
    uint32_t discontinuityCount = 0;
    for (float delta : deltaD)
    {
        squaredSum += static_cast<double>(delta) * static_cast<double>(delta);
        if (delta > JUMP_THRESHOLD)
        {
            ++discontinuityCount;
        }
    }

    result.frobeniusNorm = static_cast<float>(std::sqrt(squaredSum));
    result.discontinuityRate =
        expectedCount == 0
            ? 1.0f
            : static_cast<float>(discontinuityCount) /
                  static_cast<float>(expectedCount);
    result.pass = result.discontinuityRate < DISC_RATE_LIMIT;
    return result;
}

std::vector<EpsilonRunResult> EpsilonProbe::runPose(int poseIndex,
                                                    const glm::mat4& refView,
                                                    float epsilon)
{
    std::vector<EpsilonRunResult> results;
    results.reserve(8);
    for (int directionIndex = 0; directionIndex < 8; ++directionIndex)
    {
        results.push_back(runOne(poseIndex, refView, directionIndex, epsilon));
    }
    return results;
}

glm::mat4 EpsilonProbe::applyPerturbation(const glm::mat4& view,
                                          int directionIndex,
                                          float epsilon) const
{
    glm::mat4 perturbed = view;
    switch (directionIndex)
    {
    case 0:
        perturbed[3][0] += epsilon;
        return perturbed;
    case 1:
        perturbed[3][0] -= epsilon;
        return perturbed;
    case 2:
        perturbed[3][1] += epsilon;
        return perturbed;
    case 3:
        perturbed[3][1] -= epsilon;
        return perturbed;
    case 4:
        perturbed[3][2] += epsilon;
        return perturbed;
    case 5:
        perturbed[3][2] -= epsilon;
        return perturbed;
    case 6:
        return glm::rotate(glm::mat4(1.0f), epsilon, glm::vec3(1.0f, 0.0f, 0.0f)) *
               view;
    case 7:
        return glm::rotate(glm::mat4(1.0f), -epsilon, glm::vec3(1.0f, 0.0f, 0.0f)) *
               view;
    default:
        return view;
    }
}

std::vector<float> EpsilonProbe::computeDeltaD(const std::vector<float>& ref,
                                               const std::vector<float>& perturbed) const
{
    if (ref.size() != perturbed.size())
    {
        return {};
    }

    std::vector<float> delta(ref.size(), 0.0f);
    for (size_t i = 0; i < ref.size(); ++i)
    {
        if (ref[i] == 1.0f || perturbed[i] == 1.0f)
        {
            continue;
        }
        delta[i] = std::abs(ref[i] - perturbed[i]);
    }
    return delta;
}
