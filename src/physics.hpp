#pragma once

#include <madrona/math.hpp>

namespace madEscape {

struct Sphere {
    madrona::math::Vector3 center;
    float radius;
};

int intersectMovingSphereAABB(Sphere s, 
                              madrona::math::Vector3 dx,
                              madrona::math::AABB aabb,
                              float &t);

struct OBB {
    // Specified in clockwise order
    madrona::math::Vector2 verts[4];
};

// We aren't doing precise collision detection so we just care to see whether
// a collision has happened at all.
int intersectMovingOBBs2D(const OBB &a,
                          const OBB &b);

}
