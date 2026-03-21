#pragma once

#include "CoordinateFrame.h"

#include <glm/glm.hpp>

#include <string>
#include <vector>

struct TransformStep {
    CoordinateSystem from;
    CoordinateSystem to;
    glm::mat4 matrix;
};

struct TransformPath {
    CoordinateSystem source = CoordinateSystem::World;
    CoordinateSystem destination = CoordinateSystem::World;
    std::vector<TransformStep> steps;
    glm::mat4 combined{1.0f};
};

class TransformTree
{
public:
    static TransformTree& instance();

    void init(const glm::mat4& projectionMatrix);
    void updateProjection(const glm::mat4& projectionMatrix);

    glm::mat4 getTransform(CoordinateSystem src, CoordinateSystem dst) const;

    glm::mat4 transformPose(const glm::mat4& pose,
                            CoordinateSystem src,
                            CoordinateSystem dst) const;

    glm::vec3 transformPoint(const glm::vec3& point,
                             CoordinateSystem src,
                             CoordinateSystem dst) const;

    std::vector<TransformPath> getActivePaths() const;
    TransformPath getLastUsedPath() const;

private:
    TransformTree() = default;

    bool m_initialized = false;
    glm::mat4 m_projection{1.0f};
    glm::mat4 m_worldToCamera{1.0f};
    glm::mat4 m_cameraToWorld{1.0f};
    glm::mat4 m_cameraToNDC{1.0f};
    glm::mat4 m_worldToNDC{1.0f};

    mutable TransformPath m_lastPath;

    void rebuildCache();
    void recordPath(CoordinateSystem src,
                    CoordinateSystem dst,
                    const glm::mat4& combined) const;
};
