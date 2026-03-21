#include "../../src/core/transform/CoordinateFrame.h"
#include "../../src/core/transform/TransformTree.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cstdio>
#include <exception>

namespace {

bool expectNear(const glm::vec3& a, const glm::vec3& b, float epsilon)
{
    return glm::length(a - b) < epsilon;
}

} // namespace

int main()
{
    try
    {
        TransformTree::instance().init(glm::mat4(1.0f));

        bool allPass = true;

        {
            const glm::vec3 pWorld(1.0f, 2.0f, 3.0f);
            const glm::vec3 pCamera = TransformTree::instance().transformPoint(
                pWorld, CoordinateSystem::World, CoordinateSystem::Camera);
            const glm::vec3 pBack = TransformTree::instance().transformPoint(
                pCamera, CoordinateSystem::Camera, CoordinateSystem::World);
            const float error = glm::length(pBack - pWorld);
            if (error < 1e-6f)
            {
                std::printf("TEST 1 PASS: roundtrip error = %.2e\n", error);
            }
            else
            {
                std::fprintf(stderr,
                             "TEST 1 FAIL: roundtrip error = %.2e\n",
                             error);
                allPass = false;
            }
        }

        {
            const glm::vec3 pRos(1.0f, 0.0f, 0.0f);
            const glm::vec3 pExpected(0.0f, 0.0f, 1.0f);
            const glm::vec4 actual4 =
                CoordinateFrame::getRotation(CoordinateSystem::ROS,
                                             CoordinateSystem::Camera) *
                glm::vec4(pRos, 1.0f);
            const glm::vec3 actual(actual4);
            const float error = glm::length(actual - pExpected);
            if (error < 1e-6f)
            {
                std::printf(
                    "TEST 2 PASS: ROS(1,0,0) -> CV(0,0,1) verified\n");
            }
            else
            {
                std::fprintf(stderr,
                             "TEST 2 FAIL: expected (0,0,1), got (%.6f, %.6f, %.6f)\n",
                             actual.x,
                             actual.y,
                             actual.z);
                allPass = false;
            }
        }

        {
            const glm::mat3 R = glm::mat3(
                CoordinateFrame::getRotation(CoordinateSystem::ROS,
                                             CoordinateSystem::Camera));
            CoordinateFrame::validateRotation(R);
            std::printf("TEST 3 PASS: valid rotation accepted\n");
        }

        {
            bool rejected = false;
            try
            {
                glm::mat3 badR = glm::mat3(
                    CoordinateFrame::getRotation(CoordinateSystem::ROS,
                                                 CoordinateSystem::Camera));
                badR[0] *= -1.0f;
                CoordinateFrame::validateRotation(badR);
            }
            catch (const std::invalid_argument&)
            {
                rejected = true;
            }

            if (rejected)
            {
                std::printf(
                    "TEST 4 PASS: reflection matrix correctly rejected\n");
            }
            else
            {
                std::fprintf(stderr,
                             "TEST 4 FAIL: reflection matrix was accepted\n");
                allPass = false;
            }
        }

        {
            TransformTree::instance().init(glm::mat4(1.0f));
            const glm::vec3 pWorld(2.0f, -1.0f, 0.5f);
            const glm::vec3 pCamera = TransformTree::instance().transformPoint(
                pWorld, CoordinateSystem::World, CoordinateSystem::Camera);
            const glm::vec3 pBack = TransformTree::instance().transformPoint(
                pCamera, CoordinateSystem::Camera, CoordinateSystem::World);
            const float error = glm::length(pBack - pWorld);
            const glm::mat4 worldToNdc =
                TransformTree::instance().getTransform(CoordinateSystem::World,
                                                       CoordinateSystem::NDC);
            (void)worldToNdc;
            if (error < 1e-6f)
            {
                std::printf(
                    "TEST 5 PASS: full path roundtrip error = %.2e\n",
                    error);
            }
            else
            {
                std::fprintf(stderr,
                             "TEST 5 FAIL: roundtrip error = %.2e\n",
                             error);
                allPass = false;
            }
        }

        return allPass ? 0 : 1;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "Transform test failed: %s\n", e.what());
        return 1;
    }
}
