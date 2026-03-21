#include "TransformTree.h"

#include <glm/gtc/matrix_inverse.hpp>

#include <stdexcept>
#include <utility>

namespace {

CoordinateSystem normalizeCoordinateSystem(CoordinateSystem cs)
{
    return cs == CoordinateSystem::ROS ? CoordinateSystem::World : cs;
}

glm::mat4 applyVulkanClipTransform(const glm::mat4& projectionMatrix)
{
    glm::mat4 adjusted = projectionMatrix;
    adjusted[1][1] *= -1.0f;
    return adjusted;
}

TransformPath makeSingleStepPath(CoordinateSystem source,
                                 CoordinateSystem destination,
                                 const glm::mat4& matrix)
{
    TransformPath path{};
    path.source = source;
    path.destination = destination;
    path.combined = matrix;
    path.steps.reserve(1);
    path.steps.push_back({source, destination, matrix});
    return path;
}

TransformPath makeTwoStepPath(CoordinateSystem source,
                              CoordinateSystem middle,
                              CoordinateSystem destination,
                              const glm::mat4& first,
                              const glm::mat4& second)
{
    TransformPath path{};
    path.source = source;
    path.destination = destination;
    path.combined = second * first;
    path.steps.reserve(2);
    path.steps.push_back({source, middle, first});
    path.steps.push_back({middle, destination, second});
    return path;
}

} // namespace

TransformTree& TransformTree::instance()
{
    static TransformTree tree;
    return tree;
}

void TransformTree::init(const glm::mat4& projectionMatrix)
{
    m_projection = applyVulkanClipTransform(projectionMatrix);
    if (m_lastPath.steps.capacity() < 2)
    {
        m_lastPath.steps.reserve(2);
    }
    m_initialized = true;
    rebuildCache();
    m_lastPath = makeSingleStepPath(CoordinateSystem::World,
                                    CoordinateSystem::World,
                                    glm::mat4(1.0f));
}

void TransformTree::updateProjection(const glm::mat4& projectionMatrix)
{
    if (!m_initialized)
    {
        init(projectionMatrix);
        return;
    }

    m_projection = applyVulkanClipTransform(projectionMatrix);
    rebuildCache();
}

glm::mat4 TransformTree::getTransform(CoordinateSystem src,
                                      CoordinateSystem dst) const
{
    if (!m_initialized)
    {
        throw std::logic_error("TransformTree must be initialized before use.");
    }

    src = normalizeCoordinateSystem(src);
    dst = normalizeCoordinateSystem(dst);

    if (src == dst)
    {
        return glm::mat4(1.0f);
    }

    if (src == CoordinateSystem::World && dst == CoordinateSystem::Camera)
    {
        return m_worldToCamera;
    }
    if (src == CoordinateSystem::Camera && dst == CoordinateSystem::World)
    {
        return m_cameraToWorld;
    }
    if (src == CoordinateSystem::Camera && dst == CoordinateSystem::NDC)
    {
        return m_cameraToNDC;
    }
    if (src == CoordinateSystem::World && dst == CoordinateSystem::NDC)
    {
        return m_worldToNDC;
    }
    if (src == CoordinateSystem::NDC && dst == CoordinateSystem::Camera)
    {
        return glm::inverse(m_cameraToNDC);
    }
    if (src == CoordinateSystem::NDC && dst == CoordinateSystem::World)
    {
        return glm::inverse(m_worldToNDC);
    }

    throw std::invalid_argument(
        "TransformTree: no path from " +
        std::to_string(static_cast<int>(src)) + " to " +
        std::to_string(static_cast<int>(dst)));
}

glm::mat4 TransformTree::transformPose(const glm::mat4& pose,
                                       CoordinateSystem src,
                                       CoordinateSystem dst) const
{
    const glm::mat3 R = glm::mat3(pose);
    CoordinateFrame::validateRotation(R);

    const glm::mat4 T = getTransform(src, dst);
    const glm::mat4 result = T * pose;
    recordPath(src, dst, T);
    return result;
}

glm::vec3 TransformTree::transformPoint(const glm::vec3& point,
                                        CoordinateSystem src,
                                        CoordinateSystem dst) const
{
    const glm::mat4 T = getTransform(src, dst);
    const glm::vec4 p4(point, 1.0f);
    const glm::vec4 result = T * p4;
    return glm::vec3(result) / result.w;
}

std::vector<TransformPath> TransformTree::getActivePaths() const
{
    if (!m_initialized)
    {
        throw std::logic_error("TransformTree must be initialized before use.");
    }

    std::vector<TransformPath> paths;
    paths.reserve(4);
    paths.push_back(makeSingleStepPath(CoordinateSystem::World,
                                       CoordinateSystem::Camera,
                                       m_worldToCamera));
    paths.push_back(makeSingleStepPath(CoordinateSystem::Camera,
                                       CoordinateSystem::World,
                                       m_cameraToWorld));
    paths.push_back(makeSingleStepPath(CoordinateSystem::Camera,
                                       CoordinateSystem::NDC,
                                       m_cameraToNDC));
    paths.push_back(makeTwoStepPath(CoordinateSystem::World,
                                    CoordinateSystem::Camera,
                                    CoordinateSystem::NDC,
                                    m_worldToCamera,
                                    m_cameraToNDC));
    return paths;
}

TransformPath TransformTree::getLastUsedPath() const
{
    return m_lastPath;
}

void TransformTree::rebuildCache()
{
    m_worldToCamera = CoordinateFrame::getRotation(CoordinateSystem::World,
                                                   CoordinateSystem::Camera);
    m_cameraToWorld = CoordinateFrame::getRotation(CoordinateSystem::Camera,
                                                   CoordinateSystem::World);
    m_cameraToNDC = m_projection;
    m_worldToNDC = m_cameraToNDC * m_worldToCamera;
}

void TransformTree::recordPath(CoordinateSystem src,
                               CoordinateSystem dst,
                               const glm::mat4& combined) const
{
    const CoordinateSystem normalizedSrc = normalizeCoordinateSystem(src);
    const CoordinateSystem normalizedDst = normalizeCoordinateSystem(dst);

    m_lastPath.source = src;
    m_lastPath.destination = dst;
    m_lastPath.combined = combined;
    m_lastPath.steps.clear();

    if (normalizedSrc == normalizedDst)
    {
        m_lastPath.steps.push_back({src, dst, glm::mat4(1.0f)});
        return;
    }

    if (normalizedSrc == CoordinateSystem::World &&
        normalizedDst == CoordinateSystem::Camera)
    {
        m_lastPath.steps.push_back({src, dst, m_worldToCamera});
        return;
    }

    if (normalizedSrc == CoordinateSystem::Camera &&
        normalizedDst == CoordinateSystem::World)
    {
        m_lastPath.steps.push_back({src, dst, m_cameraToWorld});
        return;
    }

    if (normalizedSrc == CoordinateSystem::Camera &&
        normalizedDst == CoordinateSystem::NDC)
    {
        m_lastPath.steps.push_back({src, dst, m_cameraToNDC});
        return;
    }

    if (normalizedSrc == CoordinateSystem::World &&
        normalizedDst == CoordinateSystem::NDC)
    {
        m_lastPath.steps.push_back(
            {src == CoordinateSystem::ROS ? CoordinateSystem::ROS : CoordinateSystem::World,
             CoordinateSystem::Camera,
             m_worldToCamera});
        m_lastPath.steps.push_back({CoordinateSystem::Camera,
                                    dst,
                                    m_cameraToNDC});
        return;
    }

    if (normalizedSrc == CoordinateSystem::NDC &&
        normalizedDst == CoordinateSystem::Camera)
    {
        m_lastPath.steps.push_back({src, dst, glm::inverse(m_cameraToNDC)});
        return;
    }

    if (normalizedSrc == CoordinateSystem::NDC &&
        normalizedDst == CoordinateSystem::World)
    {
        m_lastPath.steps.push_back({CoordinateSystem::NDC,
                                    CoordinateSystem::Camera,
                                    glm::inverse(m_cameraToNDC)});
        m_lastPath.steps.push_back({CoordinateSystem::Camera,
                                    dst == CoordinateSystem::ROS ? CoordinateSystem::ROS : CoordinateSystem::World,
                                    m_cameraToWorld});
    }
}
