#pragma once

#include <madrona/math.hpp>

namespace madEscape {

struct Sphere {
    madrona::math::Vector3 center;
    float radius;
};

int intersectMovingSphereAABB(Sphere s, madrona::math::Vector3 dx, madrona::math::AABB aabb, float &t);

}
