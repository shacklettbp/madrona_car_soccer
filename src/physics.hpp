#pragma once

#include <madrona/math.hpp>

namespace madEscape {

struct Sphere {
    madrona::math::Vector3 center;
    float radius;
};

struct OBB {
    // Specified in clockwise order
    madrona::math::Vector2 verts[4];
};

struct WallPlane {
    madrona::math::Vector2 point;
    madrona::math::Vector2 normal;
};

struct WallSegment {
    madrona::math::Vector2 borders[2];

    // Just to know in which direction we consider a collision
    // (to make things simpler for now)
    madrona::math::Vector2 normal;
};

int intersectMovingSphereAABB(Sphere s, 
                              madrona::math::Vector3 dx,
                              madrona::math::AABB aabb,
                              float &t,
                              madrona::math::Vector3 &s_pos_out);

// We aren't doing precise collision detection so we just care to see whether
// a collision has happened at all.
int intersectMovingOBBs2D(const OBB &a,
                          const OBB &b,
                          float &min_overlap,
                          madrona::math::Vector2 &min_overlap_axis);

int intersectMovingOBBWall(const OBB &a,
                           const WallPlane &plane,
                           float &min_overlap);

#if 0
int intersectMovingSphereWall(Sphere s,
                              madrona::math::Vector3 dx,
                              const WallPlane &plane,
                              float &t_out,
                              madrona::math::Vector3 &p_out);
#endif

int intersectSphereWall(Sphere s,
                        const WallPlane &plane,
                        float &min_overlap);

int intersectSphereWallSeg(Sphere s,
                        const WallSegment &seg,
                        float &min_overlap);

}
