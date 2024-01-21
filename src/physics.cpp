#include <stdio.h>
#include "physics.hpp"

namespace madEscape {

using namespace madrona::math;

struct Segment {
    Vector3 p1;
    Vector3 p2;
};

int intersectMovingSphereAABB(Sphere s,
                              Vector3 dx,
                              AABB aabb,
                              float &t)
{
    AABB e = aabb;

    e.pMin -= Vector3::all(s.radius);
    e.pMax += Vector3::all(s.radius);

    float t_out;
    bool intersected = e.rayIntersects(s.center, 
                                          Diag3x3::fromVec(dx).inv(),
                                          0.0f, 1.0f, t_out);

    if (intersected) {
        return 1;
    } else {
        return 0;
    }
}
    
}
