#include "CoordinateFrame.h"

#include <string>

namespace {

CoordinateSystem normalizeCoordinateSystem(CoordinateSystem cs)
{
    return cs == CoordinateSystem::ROS ? CoordinateSystem::World : cs;
}

} // namespace

glm::mat4 CoordinateFrame::getRotation(CoordinateSystem src, CoordinateSystem dst)
{
    src = normalizeCoordinateSystem(src);
    dst = normalizeCoordinateSystem(dst);

    if (src == dst)
    {
        return glm::mat4(1.0f);
    }

    if (src == CoordinateSystem::World && dst == CoordinateSystem::Camera)
    {
        return buildRosToCV();
    }

    if (src == CoordinateSystem::Camera && dst == CoordinateSystem::World)
    {
        return buildCVToRos();
    }

    if (src == CoordinateSystem::NDC || dst == CoordinateSystem::NDC)
    {
        throw std::logic_error(
            "NDC projection requires explicit projection matrix, use TransformTree");
    }

    throw std::invalid_argument(
        "CoordinateFrame::getRotation unsupported transform: src=" +
        std::to_string(static_cast<int>(src)) +
        " dst=" + std::to_string(static_cast<int>(dst)));
}

glm::mat4 CoordinateFrame::transformPose(const glm::mat4& pose,
                                         CoordinateSystem src,
                                         CoordinateSystem dst)
{
    const glm::mat4 rotation = getRotation(src, dst);
    return rotation * pose;
}

void CoordinateFrame::validateRotation(const glm::mat3& R, float epsilon)
{
    const float det = determinant3x3(R);
    if (std::abs(det - 1.0f) > epsilon)
    {
        throw std::invalid_argument(
            "Invalid rotation matrix: det(R) = " + std::to_string(det) +
            ", expected +1.0 ± " + std::to_string(epsilon) +
            ". Reflection matrices are rejected at CPU boundary.");
    }
}

float CoordinateFrame::determinant3x3(const glm::mat3& m)
{
    return m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
           m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2]) +
           m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2]);
}

glm::mat4 CoordinateFrame::buildRosToCV()
{
    glm::mat4 m(0.0f);
    m[0] = glm::vec4(0.f, 0.f, 1.f, 0.f);
    m[1] = glm::vec4(-1.f, 0.f, 0.f, 0.f);
    m[2] = glm::vec4(0.f, -1.f, 0.f, 0.f);
    m[3] = glm::vec4(0.f, 0.f, 0.f, 1.f);
    return m;
}

glm::mat4 CoordinateFrame::buildCVToRos()
{
    glm::mat4 m(0.0f);
    m[0] = glm::vec4(0.f, -1.f, 0.f, 0.f);
    m[1] = glm::vec4(0.f, 0.f, -1.f, 0.f);
    m[2] = glm::vec4(1.f, 0.f, 0.f, 0.f);
    m[3] = glm::vec4(0.f, 0.f, 0.f, 1.f);
    return m;
}
