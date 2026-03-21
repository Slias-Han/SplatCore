#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <stdexcept>

enum class CoordinateSystem {
    World,
    Camera,
    NDC,
    ROS,
};

namespace CoordinateFrameConstants {

constexpr float M_ROS_TO_CV[3][3] = {
    {0.f, -1.f, 0.f},
    {0.f, 0.f, -1.f},
    {1.f, 0.f, 0.f},
};

constexpr float M_CV_TO_ROS[3][3] = {
    {0.f, 0.f, 1.f},
    {-1.f, 0.f, 0.f},
    {0.f, -1.f, 0.f},
};

} // namespace CoordinateFrameConstants

class CoordinateFrame
{
public:
    static glm::mat4 getRotation(CoordinateSystem src, CoordinateSystem dst);

    static glm::mat4 transformPose(const glm::mat4& pose,
                                   CoordinateSystem src,
                                   CoordinateSystem dst);

    static void validateRotation(const glm::mat3& R, float epsilon = 1e-4f);

    static float determinant3x3(const glm::mat3& m);

private:
    static glm::mat4 buildRosToCV();
    static glm::mat4 buildCVToRos();
};
